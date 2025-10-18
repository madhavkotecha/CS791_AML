from models import ConditionalDDPM
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse
from utils import seed_everything, compute_fid
from scheduler import NoiseSchedulerDDPM
import os
from tqdm import tqdm

def train(model, train_loader, test_loader, run_name, learning_rate, epochs, batch_size, num_steps, mask_type, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = NoiseSchedulerDDPM(num_timesteps=num_steps, type=mask_type, beta_start=1e-4, beta_end=0.02)
    scheduler.betas = scheduler.betas.to(device)
    scheduler.alphas = scheduler.alphas.to(device)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
    scheduler.sqrt_alphas_cumprod = scheduler.sqrt_alphas_cumprod.to(device)
    scheduler.sqrt_one_minus_alphas_cumprod = scheduler.sqrt_one_minus_alphas_cumprod.to(device)
    scheduler.posterior_variance = scheduler.posterior_variance.to(device)
    scheduler.alphas_cumprod_prev = scheduler.alphas_cumprod_prev.to(device)
    criterion = torch.nn.MSELoss()
    
    model.train()
    best_loss = float('inf')
    p_uncond = 0.1
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            images = images.to(device)
            labels = labels.to(device)
            batch_size_actual = images.size(0)
      
            mask = torch.rand(labels.shape, device=device) < p_uncond
            labels[mask] = -1
            
            t = torch.randint(0, scheduler.num_timesteps, (batch_size_actual,), device=device)
            
            noise = torch.randn_like(images)
            noisy_images = scheduler.q_sample(images, t, noise)
            predicted_noise = model(noisy_images, t, labels)
            
            loss = criterion(predicted_noise, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.6f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f"{run_name}/model.pth")
            print(f"Saved model with loss: {avg_loss:.6f}")

@torch.no_grad()
def sample(model, class_label, device, num_samples=16, num_steps=1000):
    model.eval()
    scheduler = NoiseSchedulerDDPM(num_timesteps=num_steps, type="linear", beta_start=1e-4, beta_end=0.02)
    
    samples = torch.randn(num_samples, 1, 28, 28, device=device)
    labels = torch.full((num_samples,), class_label, device=device, dtype=torch.long)
    
    guidance_scale = 2.0
    
    for t in tqdm(range(num_steps-1, -1, -1), desc=f"Sampling class {class_label}"):
        t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
        
        cond_noise = model(samples, t_batch, labels)
        
        uncond_labels = torch.full((num_samples,), -1, device=device, dtype=torch.long)
        uncond_noise = model(samples, t_batch, uncond_labels)

        predicted_noise = uncond_noise + guidance_scale * (cond_noise - uncond_noise)
        
        alpha_t = scheduler.alphas[t]
        sqrt_alpha_t = scheduler.sqrt_alphas[t]
        sqrt_one_minus_alpha_cumprod_t = scheduler.sqrt_one_minus_alphas_cumprod[t]
        
        samples = (1 / sqrt_alpha_t) * (samples - ((1 - alpha_t) / sqrt_one_minus_alpha_cumprod_t) * predicted_noise)
        
        if t > 0:
            posterior_variance = scheduler.posterior_variance[t]
            noise = torch.randn_like(samples)
            samples = samples + torch.sqrt(posterior_variance) * noise
    
    return torch.clamp(samples, 0, 1)

def parse_args():
    parser = argparse.ArgumentParser(description="DDPM Conditional Model Template")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_steps", type=int, default=1000, help="Number of diffusion steps")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "sample"], help="Mode: train or sample")
    parser.add_argument("--mask", type=str, default="linear", choices=["linear", "cosine"])
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Data preprocessing (do not edit)
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = ConditionalDDPM(num_classes=10, num_timesteps=args.num_steps)
    model.to(device)
    
    run_name = f"exps_conditional_ddpm/{args.epochs}ep_{args.batch_size}bs_{args.learning_rate}lr_{args.num_steps}numsteps_{args.mask}mask" 
    os.makedirs(run_name, exist_ok=True)
    
    if args.mode == "train":
        model.train()
        train(model, train_loader, test_loader, run_name, args.learning_rate, args.epochs, args.batch_size, args.num_steps, args.mask, device)
    elif args.mode == "sample":
        model.load_state_dict(torch.load(f"{run_name}/model.pth"))
        model.eval()
        for class_num in range(10):
            samples = sample(model, class_num, device, args.num_samples, args.num_steps)
            torch.save(samples, f"{run_name}/{class_num}class_{args.num_samples}samples_{args.num_steps}steps.pt")
