import torch
import math
import numpy as np

class NoiseSchedulerDDPM():
    def __init__(self, num_timesteps=1000, type="linear", **kwargs):
        self.num_timesteps = num_timesteps
        self.type = type
        
        if type == "linear":
            self.init_linear_schedule(**kwargs)
        elif type == "cosine":
            self.init_cosine_schedule(**kwargs)
        else:
            raise NotImplementedError(f"{type} scheduler is not implemented")
            
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-20)
        
    def init_linear_schedule(self, beta_start=1e-4, beta_end=0.02):
        self.betas = torch.linspace(beta_start, beta_end, self.num_timesteps, dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
    def init_cosine_schedule(self, beta_start=1e-4, beta_end=0.02, s=0.008):
        def cosine_beta_schedule(timesteps, s=0.008):
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clamp(betas, 0, 0.999)
        
        self.betas = cosine_beta_schedule(self.num_timesteps, s)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        
    def __len__(self):
        return self.num_timesteps


class MaskSchedulerD3PM():
    """
    Mask scheduler for Discrete Diffusion (D3PM) models.

    Args:
        num_timesteps: int, number of timesteps in the diffusion process
        mask_type: str, type of mask scheduling ("uniform", "linear", etc.)
        **kwargs: additional arguments for mask scheduling

    This object sets up the mask schedule for each timestep.
    """

    def __init__(self, num_timesteps=50, mask_type="uniform", **kwargs):
        self.num_timesteps = num_timesteps
        self.mask_type = mask_type

        if mask_type == "linear":
            self.init_linear_schedule(**kwargs)
        elif mask_type =="cosine":
            self.init_cosine_schedule(**kwargs)
        else:
            raise NotImplementedError(f"{mask_type} mask scheduler is not implemented")

    def init_linear_schedule(self, mask_start=0.0, mask_end=0.95):
        """
        Initializes a linear mask schedule where the mask probability increases linearly.
        """
        self.mask_probs = torch.linspace(mask_start, mask_end, self.num_timesteps, dtype=torch.float32)
        
        self.mask_token = 256

    def init_cosine_schedule(self,  mask_start=0.0, mask_end=0.95):
        # theorem 2. from paper https://arxiv.org/pdf/2508.04884
        x =  torch.linspace(0, self.num_timesteps, self.num_timesteps+1, dtype=torch.float64) 
        alpha_t = torch.cos((x / self.num_timesteps) *( torch.pi / 2)) ** 2
        self.mask_probs = 1.0 - alpha_t
        self.mask_token = 256


    def add_mask(self, x, timesteps):
        batch_size = x.shape[0]
        device = x.device
        
        mask_probs_t = self.mask_probs[timesteps]
        
        random_mask = torch.rand(x.shape, device=device)
        
        mask_condition = random_mask < mask_probs_t.reshape(-1, 1, 1)
        
        masked_x = x.clone()
        masked_x[mask_condition] = self.mask_token
        
        return masked_x
    
    def step(self, model_output, timestep, sample):
        device = sample.device
        
        current_mask_prob = self.mask_probs[timestep] if timestep < len(self.mask_probs) else 0.0
        prev_mask_prob = self.mask_probs[timestep - 1] if timestep > 0 else 0.0
        
        currently_masked = (sample == self.mask_token)
        
        if model_output.dim() == 4:
            probs = torch.softmax(model_output, dim=-1)
            sampled_pixels = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(sample.shape)
        else:
            sampled_pixels = model_output
        
        next_sample = sample.clone()
        
        if timestep > 0:
            unmask_prob = (current_mask_prob - prev_mask_prob) / current_mask_prob if current_mask_prob > 0 else 0.0
            
            random_unmask = torch.rand(sample.shape, device=device)
            should_unmask = (random_unmask < unmask_prob) & currently_masked
            
            next_sample[should_unmask] = sampled_pixels[should_unmask]
        else:
            next_sample[currently_masked] = sampled_pixels[currently_masked]
        
        return next_sample

    def __len__(self):
        return self.num_timesteps