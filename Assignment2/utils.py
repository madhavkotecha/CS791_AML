import numpy as np
import torch
import random
from torchmetrics.image.fid import FrechetInceptionDistance
import torch.nn.functional as F

def seed_everything(seed):
    # https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial3/Activation_Functions.html
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available(): # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    random.seed(seed)

def compute_fid(real_images, fake_images):
    '''
        Args:
            real_images (Actual images from the dataset): torch.Tensor, shape (N, 1, 28, 28), range [0, 1]
            fake_images (Generated images by the diffusion model): torch.Tensor, shape (N, 1, 28, 28), range [0, 1]
        Returns:
            fid (Frechet Inception Distance): torch.Tensor
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fid = FrechetInceptionDistance().to(device)
    # Move images to device
    real_images = real_images.to(device)
    fake_images = fake_images.to(device)
    # Convert to uint8 and [0, 255]
    real_images = (real_images * 255).clamp(0, 255).to(torch.uint8)
    fake_images = (fake_images * 255).clamp(0, 255).to(torch.uint8)
    # Convert 1 channel to 3 channels
    real_images = real_images.repeat(1, 3, 1, 1)
    fake_images = fake_images.repeat(1, 3, 1, 1)
    # Resize to 299x299
    real_images = F.interpolate(real_images.float(), size=(299, 299), mode='nearest').to(device)
    fake_images = F.interpolate(fake_images.float(), size=(299, 299), mode='nearest').to(device)
    # Convert back to uint8
    real_images = real_images.clamp(0, 255).to(torch.uint8)
    fake_images = fake_images.clamp(0, 255).to(torch.uint8)
    # Update FID metric
    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)
    return fid.compute()



#---Impl. without torchmetrics---#

# def compute_fid(real_images, fake_images):
#     '''
#         Args:
#             real_images (Actual images from the dataset): torch.Tensor, shape (N, 1, 28, 28), range [0, 1]
#             fake_images (Generated images by the diffusion model): torch.Tensor, shape (N, 1, 28, 28), range [0, 1]
#         Returns:
#             fid (Frechet Inception Distance): torch.Tensor
#     '''
#     try:
#         if real_images.shape != fake_images.shape:
#             raise ValueError(f"Shape mismatch: real_images {real_images.shape} vs fake_images {fake_images.shape}")
        
#         if len(real_images.shape) != 4 or real_images.shape[1] != 1:
#             raise ValueError(f"Expected shape (N, 1, H, W), got {real_images.shape}")
        
#         if real_images.min() < 0 or real_images.max() > 1:
#             print(f"Warning: real_images values outside [0,1] range: [{real_images.min():.3f}, {real_images.max():.3f}]")
#         if fake_images.min() < 0 or fake_images.max() > 1:
#             print(f"Warning: fake_images values outside [0,1] range: [{fake_images.min():.3f}, {fake_images.max():.3f}]")
        
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
#         batch_size = real_images.shape[0]
#         if batch_size > 100 and device.type == 'cuda':
#             print(f"Large batch size ({batch_size}), processing in chunks to avoid memory issues")
#             chunk_size = 50
#             fid_scores = []
            
#             for i in range(0, batch_size, chunk_size):
#                 end_idx = min(i + chunk_size, batch_size)
#                 real_chunk = real_images[i:end_idx]
#                 fake_chunk = fake_images[i:end_idx]
                
#                 chunk_fid = _compute_fid_chunk(real_chunk, fake_chunk, device)
#                 fid_scores.append(chunk_fid)
            
#             return torch.mean(torch.stack(fid_scores))
#         else:
#             return _compute_fid_chunk(real_images, fake_images, device)
            
#     except Exception as e:
#         print(f"Error computing FID: {e}")
#         raise

# def _compute_fid_chunk(real_images, fake_images, device):
#     try:
#         from torchmetrics.image.fid import FrechetInceptionDistance
#         fid = FrechetInceptionDistance().to(device)
#     except ImportError:
#         print("Warning: torchmetrics not available. Using simplified FID computation...")
#         return _compute_simplified_fid(real_images, fake_images, device)
#     except Exception as e:
#         print(f"Warning: Could not initialize FrechetInceptionDistance: {e}")
#         print("Falling back to simplified FID computation...")
#         return _compute_simplified_fid(real_images, fake_images, device)
    
#     try:
#         real_images = real_images.to(device)
#         fake_images = fake_images.to(device)
#     except RuntimeError as e:
#         if "out of memory" in str(e).lower():
#             print("CUDA out of memory. Falling back to CPU.")
#             device = torch.device("cpu")
#             from torchmetrics.image.fid import FrechetInceptionDistance
#             fid = FrechetInceptionDistance().to(device)
#             real_images = real_images.to(device)
#             fake_images = fake_images.to(device)
#         else:
#             raise
    
#     real_images = (real_images * 255).clamp(0, 255).to(torch.uint8)
#     fake_images = (fake_images * 255).clamp(0, 255).to(torch.uint8)
    
#     real_images = real_images.repeat(1, 3, 1, 1)
#     fake_images = fake_images.repeat(1, 3, 1, 1)
    
#     real_images = F.interpolate(real_images.float(), size=(299, 299), mode='bilinear', align_corners=False)
#     fake_images = F.interpolate(fake_images.float(), size=(299, 299), mode='bilinear', align_corners=False)
    
#     real_images = real_images.clamp(0, 255).to(torch.uint8)
#     fake_images = fake_images.clamp(0, 255).to(torch.uint8)
    
#     fid.update(real_images, real=True)
#     fid.update(fake_images, real=False)
    
#     return fid.compute()

# def _compute_simplified_fid(real_images, fake_images, device):
#     real_images = real_images.to(device)
#     fake_images = fake_images.to(device)
    
#     real_flat = real_images.view(real_images.size(0), -1).float()
#     fake_flat = fake_images.view(fake_images.size(0), -1).float()
    
#     mu_real = torch.mean(real_flat, dim=0)
#     mu_fake = torch.mean(fake_flat, dim=0)
    
#     mean_diff = torch.mean((mu_real - mu_fake) ** 2)
    
#     var_real = torch.var(real_flat, dim=0)
#     var_fake = torch.var(fake_flat, dim=0)
#     var_diff = torch.mean((var_real - var_fake) ** 2)
    
#     simplified_fid = mean_diff + var_diff
    
#     print(f"Note: Using simplified FID computation (not true FID)")
#     return simplified_fid