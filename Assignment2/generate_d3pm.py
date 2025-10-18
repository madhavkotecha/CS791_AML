import torch
import os
import argparse
from models import DDPM, D3PM, ConditionalDDPM, ConditionalD3PM
from scheduler import NoiseSchedulerDDPM, MaskSchedulerD3PM
from utils import seed_everything


def sample_ddpm(model, device, num_samples=64, num_steps=1000):
    """Sample from unconditional DDPM model"""
    model.eval()
    scheduler = NoiseSchedulerDDPM(num_timesteps=num_steps, type="linear", beta_start=1e-4, beta_end=2e-2)
    
    # Move scheduler to device
    scheduler.betas = scheduler.betas.to(device)
    scheduler.alphas = scheduler.alphas.to(device)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
    scheduler.sqrt_alphas_cumprod = scheduler.sqrt_alphas_cumprod.to(device)
    scheduler.sqrt_one_minus_alphas_cumprod = scheduler.sqrt_one_minus_alphas_cumprod.to(device)
    scheduler.sqrt_recip_alphas = scheduler.sqrt_recip_alphas.to(device)
    scheduler.sqrt_recipm1_alphas_cumprod = scheduler.sqrt_recipm1_alphas_cumprod.to(device)
    scheduler.posterior_variance = scheduler.posterior_variance.to(device)
    scheduler.alphas_cumprod_prev = scheduler.alphas_cumprod_prev.to(device)
    
    x = torch.randn(num_samples, 1, 28, 28, device=device)
    
    with torch.no_grad():
        for t in reversed(range(num_steps)):
            timesteps = torch.full((num_samples,), t, device=device, dtype=torch.long)
            predicted_noise = model(x, timesteps)
            x = scheduler.step(predicted_noise, t, x)
    
    return torch.clamp(x, 0.0, 1.0)


def sample_conditional_ddpm(model, device, class_label, num_samples=64, num_steps=1000):
    """Sample from conditional DDPM model for a specific class"""
    model.eval()
    scheduler = NoiseSchedulerDDPM(num_timesteps=num_steps, type="linear", beta_start=1e-4, beta_end=2e-2)
    
    # Move scheduler to device
    scheduler.betas = scheduler.betas.to(device)
    scheduler.alphas = scheduler.alphas.to(device)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
    scheduler.sqrt_alphas_cumprod = scheduler.sqrt_alphas_cumprod.to(device)
    scheduler.sqrt_one_minus_alphas_cumprod = scheduler.sqrt_one_minus_alphas_cumprod.to(device)
    scheduler.sqrt_recip_alphas = scheduler.sqrt_recip_alphas.to(device)
    scheduler.sqrt_recipm1_alphas_cumprod = scheduler.sqrt_recipm1_alphas_cumprod.to(device)
    scheduler.posterior_variance = scheduler.posterior_variance.to(device)
    scheduler.alphas_cumprod_prev = scheduler.alphas_cumprod_prev.to(device)
    
    x = torch.randn(num_samples, 1, 28, 28, device=device)
    class_labels = torch.full((num_samples,), class_label, device=device, dtype=torch.long)
    
    with torch.no_grad():
        for t in reversed(range(num_steps)):
            timesteps = torch.full((num_samples,), t, device=device, dtype=torch.long)
            predicted_noise = model(x, timesteps, class_labels)
            x = scheduler.step(predicted_noise, t, x)
    
    return torch.clamp(x, 0.0, 1.0)


def sample_d3pm(model, device, num_samples=64, num_steps=1000, mask="linear"):
    """Sample from unconditional D3PM model"""
    model.eval()
    print("mask and num_steps ", mask, num_steps)
    scheduler = MaskSchedulerD3PM(num_timesteps=num_steps, mask_type=mask, mask_start=0.0, mask_end=0.95)
    scheduler.mask_probs = scheduler.mask_probs.to(device)
    
    x = torch.full((num_samples, 28, 28), scheduler.mask_token, device=device, dtype=torch.long)
    
    with torch.no_grad():
        for t in reversed(range(num_steps)):
            timesteps = torch.full((num_samples,), t, device=device, dtype=torch.long)
            predicted_logits = model(x, timesteps)
            predicted_logits_transposed = predicted_logits.permute(0, 2, 3, 1)
            x = scheduler.step(predicted_logits_transposed, t, x)
    
    x = torch.clamp(x, 0, 255)
    x_continuous = x.float() / 255.0
    return x_continuous.unsqueeze(1)


def sample_conditional_d3pm(model, device, class_label, num_samples=64, num_steps=1000, mask="linear"):
    """Sample from conditional D3PM model for a specific class"""
    model.eval()
    print("mask and num_steps ", mask, num_steps)
    scheduler = MaskSchedulerD3PM(num_timesteps=num_steps, mask_type=mask, mask_start=0.0, mask_end=0.95)
    scheduler.mask_probs = scheduler.mask_probs.to(device)
    
    x = torch.full((num_samples, 28, 28), scheduler.mask_token, device=device, dtype=torch.long)
    class_labels = torch.full((num_samples,), class_label, device=device, dtype=torch.long)
    
    with torch.no_grad():
        for t in reversed(range(num_steps)):
            timesteps = torch.full((num_samples,), t, device=device, dtype=torch.long)
            predicted_logits = model(x, timesteps, class_labels)
            predicted_logits_transposed = predicted_logits.permute(0, 2, 3, 1)
            x = scheduler.step(predicted_logits_transposed, t, x)
    
    x = torch.clamp(x, 0, 255)
    x_continuous = x.float() / 255.0
    return x_continuous.unsqueeze(1)


def load_model(model_path, model_type, device, **kwargs):
    """Load a trained model from checkpoint"""
    if model_type == 'ddpm':
        model = DDPM(**kwargs)
    elif model_type == 'ddpm_cond':
        model = ConditionalDDPM(**kwargs)
    elif model_type == 'd3pm':
        model = D3PM(**kwargs)
    elif model_type == 'd3pm_cond':
        model = ConditionalD3PM(**kwargs)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ddpm_path", default="models/ddpm_model.pth")
    parser.add_argument("--ddpm_cond_path", default="models/ddpm_cond_model.pth")
    parser.add_argument("--d3pm_path", default="models/d3pm_model.pth")
    parser.add_argument("--d3pm_cond_path", default="models/d3pm_cond_model.pth")
    parser.add_argument("--output_dir", default="generated_samples")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--sampling_steps", type=int, default=1000)
    args = parser.parse_args()
    # import pdb; pdb.set_trace()
    mask = "linear"
    if "cosine" in args.d3pm_path or "cosine" in args.d3pm_cond_path:
        mask = "cosine"

    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    seed_everything(42)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Generating samples on {device}...")
    
    # Generate DDPM samples
    if os.path.exists(args.ddpm_path):
        print("Generating DDPM samples...")
        model = load_model(args.ddpm_path, 'ddpm', device, 
                          hidden_dim=args.hidden_dim, num_timesteps=args.sampling_steps,
                          in_channels=1, out_channels=1)
        samples = sample_ddpm(model, device, args.num_samples, args.sampling_steps)
        filename = f'samples_ddpm_numsteps{args.num_steps}.pt'
        torch.save(samples, os.path.join(args.output_dir, filename))
        print(f"✓ Saved samples_ddpm.pt {samples.shape}")
        del model
    
    # Generate conditional DDPM samples
    if os.path.exists(args.ddpm_cond_path):
        print("Generating conditional DDPM samples...")
        model = load_model(args.ddpm_cond_path, 'ddpm_cond', device,
                          hidden_dim=args.hidden_dim, num_timesteps=args.num_steps,
                          in_channels=1, out_channels=1, num_classes=10)
        for class_label in range(10):
            samples = sample_conditional_ddpm(model, device, class_label, args.num_samples, args.sampling_steps)
            filename = f'samples_ddpm_cond_{class_label}_numsteps{args.num_steps}.pt'
            torch.save(samples, os.path.join(args.output_dir, filename))
            print(f"✓ Saved {filename} {samples.shape}")
        del model
    
    # Generate D3PM samples
    if os.path.exists(args.d3pm_path):
        print("Generating D3PM samples...")
        model = load_model(args.d3pm_path, 'd3pm', device,
                          vocab_size=256, hidden_dim=args.hidden_dim, num_timesteps=args.num_steps)
        samples = sample_d3pm(model, device, args.num_samples, args.sampling_steps, mask)
        filename = f'samples_d3pm.pt'
        torch.save(samples, os.path.join(args.output_dir, filename))
        print(f"✓ Saved samples_d3pm.pt {samples.shape}")
        del model
    
    # Generate conditional D3PM samples
    if os.path.exists(args.d3pm_cond_path):
        print("Generating conditional D3PM samples...")
        model = load_model(args.d3pm_cond_path, 'd3pm_cond', device,
                          vocab_size=256, hidden_dim=args.hidden_dim, 
                          num_timesteps=args.num_steps, num_classes=10)
        for class_label in range(10):
            samples = sample_conditional_d3pm(model, device, class_label, args.num_samples, args.sampling_steps, mask)
            filename = f'samples_d3pm_cond_{class_label}.pt'
            torch.save(samples, os.path.join(args.output_dir, filename))
            print(f"✓ Saved {filename} {samples.shape}")
        del model
    
    print("Sample generation completed!")


if __name__ == "__main__":
    main()