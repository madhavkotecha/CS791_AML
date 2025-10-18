import os, argparse, json
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils import compute_fid, seed_everything
from tqdm import tqdm

def load_test_images(num_images=None):
    transform = transforms.Compose([transforms.ToTensor()])
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    loader = DataLoader(testset, batch_size=256, shuffle=False)

    imgs, labels = [], []
    for x, y in loader:
        imgs.append(x)
        labels.append(y)
        if num_images is not None and len(torch.cat(imgs)) >= num_images:
            break
    imgs = torch.cat(imgs)  # (N,1,28,28)
    labels = torch.cat(labels)
    if num_images is not None and imgs.shape[0] > num_images:
        imgs, labels = imgs[:num_images], labels[:num_images]

    return imgs, labels


def compute_fid_with_resampling(fake_imgs, real_imgs, num_subsets=5, subset_size=64):
    fids = []
    Nf, Nr = fake_imgs.shape[0], real_imgs.shape[0]
    for _ in tqdm(range(num_subsets)):
        idx_fake = torch.randperm(Nf)[:subset_size]
        idx_real = torch.randperm(Nr)[:subset_size]
        f_subset = fake_imgs[idx_fake]
        r_subset = real_imgs[idx_real]
        fids.append(compute_fid(r_subset, f_subset))  # from utils.py
    return torch.stack(fids).mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_dir", type=str, default="generated_samples")
    parser.add_argument("--ref_images", type=int, default=10000)
    parser.add_argument("--num_subsets", type=int, default=10)
    parser.add_argument("--subset_size", type=int, default=64)
    args = parser.parse_args()
    
    seed_everything(30)

    # load sample of test images
    real_imgs, real_labels = load_test_images(args.ref_images)

    results = {}
    path_uncond = os.path.join(args.generated_dir, "samples_d3pm.pt")
    if os.path.exists(path_uncond):
        print("Unconditional D3PM")
        fake_imgs = torch.load(path_uncond).cpu()
        fid = compute_fid_with_resampling(fake_imgs, real_imgs,
                                          num_subsets=args.num_subsets, subset_size=args.subset_size)
        print(f"[D3PM Unconditional] FID: {fid:.4f}")

    per_class = {}
    for cls in range(10):
        print("Conditional D3PM for class ", cls)
        path_cond = os.path.join(args.generated_dir, f"samples_d3pm_cond_{cls}.pt")
        if os.path.exists(path_cond):
            fake_imgs = torch.load(path_cond).cpu()
            real_cls_imgs = real_imgs[real_labels == cls]
            fid = compute_fid_with_resampling(fake_imgs, real_cls_imgs,
                                              num_subsets=args.num_subsets, subset_size=args.subset_size)
            per_class[str(cls)] = fid
            print(f"[D3PM Conditional] Class {cls} FID: {fid:.4f}")

    if per_class:
        avg_fid = sum(per_class.values()) / len(per_class)
        print(f"[D3PM Conditional] Avg FID: {avg_fid:.4f}")

    '''
        Evaluation of DDPM
    '''
    path_ddpm_uncond = os.path.join(args.generated_dir, "samples_ddpm.pt")
    if os.path.exists(path_ddpm_uncond):
        fake_imgs = torch.load(path_ddpm_uncond).cpu()
        fid = compute_fid_with_resampling(fake_imgs, real_imgs, num_subsets=args.num_subsets, subset_size=args.subset_size)
        print(f"[DDPM Unconditional] FID: {fid:.4f}")

    ddpm_per_class = {}
    for cls in range(10):
        path_ddpm_cond = os.path.join(args.generated_dir, f"samples_ddpm_cond_{cls}.pt")
        if os.path.exists(path_ddpm_cond):
            fake_imgs = torch.load(path_ddpm_cond).cpu()
            real_cls_imgs = real_imgs[real_labels == cls]
            fid = compute_fid_with_resampling(fake_imgs, real_cls_imgs,
                                              num_subsets=args.num_subsets, subset_size=args.subset_size)
            ddpm_per_class[str(cls)] = fid
            print(f"[DDPM Conditional] Class {cls} FID: {fid:.4f}")

    if len(ddpm_per_class) > 0:
        avg_fid = sum(ddpm_per_class.values()) / len(ddpm_per_class)
        print(f"[DDPM Conditional] Avg FID: {avg_fid:.4f}")

if __name__ == "__main__":
    main()
