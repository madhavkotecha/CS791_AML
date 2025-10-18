import torch
import os
import argparse
from models import DDPM, ConditionalDDPM
from scheduler import NoiseSchedulerDDPM
from utils import seed_everything
from tqdm import tqdm

def sample_ddpm(model, device, num_samples=64, num_steps=1000, mask="linear"):
    """Sample from unconditional DDPM model"""
    model.eval()
    scheduler = NoiseSchedulerDDPM(num_timesteps=num_steps, type=mask, beta_start=1e-4, beta_end=2e-2)
    print("MASK" ,mask)
    # Move scheduler tensors to device
    scheduler.betas = scheduler.betas.to(device)
    scheduler.alphas = scheduler.alphas.to(device)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
    scheduler.sqrt_alphas_cumprod = scheduler.sqrt_alphas_cumprod.to(device)
    scheduler.sqrt_one_minus_alphas_cumprod = scheduler.sqrt_one_minus_alphas_cumprod.to(device)
    scheduler.posterior_variance = scheduler.posterior_variance.to(device)
    scheduler.alphas_cumprod_prev = scheduler.alphas_cumprod_prev.to(device)
    scheduler.sqrt_alphas = scheduler.sqrt_alphas.to(device)
    
    x = torch.randn(num_samples, 1, 28, 28, device=device)
    
    with torch.no_grad():
        for t in tqdm(range(num_steps-1, -1, -1), desc="Sampling"):
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = model(x, t_batch)
            
            # Compute denoised sample
            alpha_t = scheduler.alphas[t]
            sqrt_alpha_t = scheduler.sqrt_alphas[t]
            sqrt_one_minus_alpha_cumprod_t = scheduler.sqrt_one_minus_alphas_cumprod[t]
            
            # Compute mean
            x = (1 / sqrt_alpha_t) * (x - ((1 - alpha_t) / sqrt_one_minus_alpha_cumprod_t) * predicted_noise)
            
            # Add noise (except for last step)
            if t > 0:
                posterior_variance = scheduler.posterior_variance[t]
                noise = torch.randn_like(x)
                x = x + torch.sqrt(posterior_variance) * noise
    
    return torch.clamp(x, 0.0, 1.0)

def sample_conditional_ddpm(model, device, class_label, num_samples=64, num_steps=1000,  mask="linear"):
    """Sample from conditional DDPM model for a specific class"""
    model.eval()
    print()
    print("MASK" ,mask, " num_steps ", num_steps)
    scheduler = NoiseSchedulerDDPM(num_timesteps=num_steps, type=mask, beta_start=1e-4, beta_end=2e-2)
 
    scheduler.betas = scheduler.betas.to(device)
    scheduler.alphas = scheduler.alphas.to(device)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
    scheduler.sqrt_alphas_cumprod = scheduler.sqrt_alphas_cumprod.to(device)
    scheduler.sqrt_one_minus_alphas_cumprod = scheduler.sqrt_one_minus_alphas_cumprod.to(device)
    scheduler.posterior_variance = scheduler.posterior_variance.to(device)
    scheduler.alphas_cumprod_prev = scheduler.alphas_cumprod_prev.to(device)
    scheduler.sqrt_alphas = scheduler.sqrt_alphas.to(device)
    
    x = torch.randn(num_samples, 1, 28, 28, device=device)
    class_labels = torch.full((num_samples,), class_label, device=device, dtype=torch.long)
    
   
    guidance_scale = 2.0
    
    with torch.no_grad():
        for t in tqdm(range(num_steps-1, -1, -1), desc=f"Sampling class {class_label}"):
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
            
            # Conditional prediction
            cond_noise = model(x, t_batch, class_labels)
            
            # Unconditional prediction
            uncond_labels = torch.full((num_samples,), -1, device=device, dtype=torch.long)
            uncond_noise = model(x, t_batch, uncond_labels)
            
            # Classifier-free guidance
            predicted_noise = uncond_noise + guidance_scale * (cond_noise - uncond_noise)
            
            # Compute denoised sample
            alpha_t = scheduler.alphas[t]
            sqrt_alpha_t = scheduler.sqrt_alphas[t]
            sqrt_one_minus_alpha_cumprod_t = scheduler.sqrt_one_minus_alphas_cumprod[t]
            
            # Compute mean
            x = (1 / sqrt_alpha_t) * (x - ((1 - alpha_t) / sqrt_one_minus_alpha_cumprod_t) * predicted_noise)
            
            # Add noise (except for last step)
            if t > 0:
                posterior_variance = scheduler.posterior_variance[t]
                noise = torch.randn_like(x)
                x = x + torch.sqrt(posterior_variance) * noise
    
    return torch.clamp(x, 0.0, 1.0)

def load_model(model_path, model_type, device, num_steps, **kwargs):
    """Load a trained model from checkpoint"""
    if model_type == 'ddpm':
        print("num_steps, ", num_steps)
        model = DDPM(num_timesteps=num_steps)
    elif model_type == 'ddpm_cond':
        model = ConditionalDDPM(num_classes=kwargs.get('num_classes', 10), num_timesteps=num_steps)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ddpm_path", default="exps_ddpm/10ep_64bs_0.0001lr/model.pth")
    parser.add_argument("--ddpm_cond_path", default="exps_conditional_ddpm/10ep_64bs_0.0001lr/model.pth")
    parser.add_argument("--output_dir", default="generated_samples")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mask", default="linear")
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--sampling_steps", type=int, default=1000)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    seed_everything(42)
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Generating samples on {device}...")
    
    mask = args.mask
    if "cosine" in args.ddpm_path or "cosine" in args.ddpm_cond_path:
        mask = "cosine"
    # Generate DDPM samples
    if os.path.exists(args.ddpm_path):
        print("Generating DDPM samples...")
        model = load_model(args.ddpm_path, 'ddpm', device, args.sampling_steps)
        samples = sample_ddpm(model, device, args.num_samples, args.sampling_steps, mask)
        torch.save(samples, os.path.join(args.output_dir, 'samples_ddpm.pt'))
        print(f"✓ Saved samples_ddpm.pt {samples.shape}")
        del model
        torch.cuda.empty_cache()
    
    # Generate conditional DDPM samples
    if os.path.exists(args.ddpm_cond_path):
        print(f"Generating conditional DDPM samples...{args.ddpm_cond_path}")
        model = load_model(args.ddpm_cond_path, 'ddpm_cond', device, args.sampling_steps, num_classes=10)
        
        for class_label in range(10):
            samples = sample_conditional_ddpm(model, device, class_label, args.num_samples, args.sampling_steps, mask)
            filename = f'samples_ddpm_cond_{class_label}.pt'
            torch.save(samples, os.path.join(args.output_dir, filename))
            print(f"✓ Saved {filename} {samples.shape}")
        
        del model
        torch.cuda.empty_cache()
    
    print("Sample generation completed!")

if __name__ == "__main__":
    main()
