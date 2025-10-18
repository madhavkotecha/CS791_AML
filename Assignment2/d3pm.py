from models import D3PM
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse
from utils import seed_everything, compute_fid
from scheduler import MaskSchedulerD3PM
import os

def train(model, train_loader, test_loader, run_name, learning_rate, epochs, batch_size, device, num_timesteps, mask):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    scheduler = MaskSchedulerD3PM(num_timesteps=num_timesteps, mask_type=mask, mask_start=0.0, mask_end=0.95)
    scheduler.mask_probs = scheduler.mask_probs.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    
    best_loss = 1e5
    best_epoch = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)
            batch_size_actual = images.shape[0]
            
            discrete_images = (images * 255).long().squeeze(1)
            discrete_images = torch.clamp(discrete_images, 0, 255)
            
            timesteps = torch.randint(0, scheduler.num_timesteps, (batch_size_actual,), device=device)
            
            masked_images = scheduler.add_mask(discrete_images, timesteps)
            
            predicted_logits = model(masked_images, timesteps)
            
            predicted_logits_flat = predicted_logits.permute(0, 2, 3, 1).contiguous().view(-1, predicted_logits.shape[1])
            discrete_images_flat = discrete_images.view(-1)
            
            loss = criterion(predicted_logits_flat, discrete_images_flat)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.6f}')
        
        avg_loss = total_loss / num_batches
        print(f'Epoch [{epoch+1}/{epochs}] completed. Average Loss: {avg_loss:.6f}')
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
        checkpoint_path = os.path.join(run_name, f'model_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f'Model checkpoint saved to {checkpoint_path}')
        print(f"best loss: {best_loss} for epoch: {best_epoch}")
    final_model_path = os.path.join(run_name, 'model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f'Final model saved to {final_model_path}')
    
    print('Training completed!')

def sample(model, device, num_samples=16, num_steps=1000):
    '''
    Returns:
        torch.Tensor, shape (num_samples, 1, 28, 28)
    '''
    model.eval()
    
    scheduler = MaskSchedulerD3PM(num_timesteps=1000, mask_type="linear", mask_start=0.0, mask_end=0.95)
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
    
    x_continuous = x_continuous.unsqueeze(1)
    
    return x_continuous

def parse_args():
    parser = argparse.ArgumentParser(description="D3PM Model")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_steps", type=int, default=1000, help="Number of diffusion steps")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "sample"], help="Mode: train or sample")
    parser.add_argument("--mask", type=str, default="linear", choices=["cosine", "linear"])
    # Add any other arguments you want here
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    ### Data Preprocessing Start ### (Do not edit this)
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    ### Data Preprocessing End ### (Do not edit this)

    model = D3PM()
    model.to(device)

    run_name = f"exps_d3pm/{args.epochs}ep_{args.batch_size}bs_{args.learning_rate}lr_{args.num_steps}numsteps_{args.mask}mask" 
    os.makedirs(run_name, exist_ok=True)

    if args.mode == "train":
        model.train()
        train(model, train_loader, test_loader, run_name, args.learning_rate, args.epochs, args.batch_size, device, args.num_steps, args.mask)
    elif args.mode == "sample":
        model.load_state_dict(torch.load(f"{run_name}/model.pth"))
        model.eval()
        samples = sample(model, device, args.num_samples, args.num_steps)
        torch.save(samples, f"{run_name}/{args.num_samples}samples_{args.num_steps}steps.pt")
