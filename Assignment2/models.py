import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TimeEmbedding(nn.Module):
    def __init__(self, time_dim=128):
        super().__init__()
        self.time_embedding = nn.Sequential(
            nn.Linear(1, 16),
            nn.Linear(16, time_dim),
            nn.ReLU()
        )
    
    def forward(self, t):
        return self.time_embedding(t.view(-1, 1))

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, time_dim=128, num_classes=0):
        super().__init__()
        self.time_dim = time_dim
        self.num_classes = num_classes
        
        self.time_mlp = TimeEmbedding(time_dim)
        
        if num_classes > 0:
            self.class_emb = nn.Embedding(num_classes, time_dim)
        
        self.input_conv = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.enc1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU()
        )
        
        self.middle = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.GroupNorm(8, 512),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU()
        )
        
        self.time_proj1 = nn.Linear(time_dim, 64)
        self.time_proj2 = nn.Linear(time_dim, 128)
        self.time_proj3 = nn.Linear(time_dim, 256)

        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU()
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU()
        )
        
        self.output = nn.Conv2d(64, out_channels, 1)
    
    def forward(self, x, t, y=None):
        t_emb = self.time_mlp(t)
        
        if y is not None and self.num_classes > 0:
            y_mask = (y != -1)
            y_emb = torch.zeros_like(t_emb)
            if y_mask.any():
                y_emb[y_mask] = self.class_emb(y[y_mask])
            t_emb = t_emb + y_emb
        
        h = self.input_conv(x)
        
        h1 = self.enc1(h)
        h1 = h1 + self.time_proj1(t_emb)[:, :, None, None]
        h2 = self.enc2(h1)
        h2 = h2 + self.time_proj2(t_emb)[:, :, None, None]
        h3 = self.enc3(h2)
        h3 = h3 + self.time_proj3(t_emb)[:, :, None, None]
        
        h_mid = self.middle(h3)
        
        h = self.dec3(torch.cat([h_mid, h3], dim=1))
        h = self.dec2(torch.cat([h, h2], dim=1))
        h = self.dec1(torch.cat([h, h1], dim=1))
        
        return self.output(h)

class TimestepEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        
    def forward(self, timesteps):
        device = timesteps.device
        half_dim = self.embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, time_emb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = nn.ReLU()(h)
        
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = nn.ReLU()(h)
        
        return h + self.shortcut(x)

class DDPM(nn.Module):
    def __init__(self,  num_timesteps=1000, num_classes=10,):
        super().__init__()
        self.unet = UNet(in_channels=1, out_channels=1, time_dim=128, num_classes=0)
        self.num_timesteps = num_timesteps
    
    def forward(self, x, t):
        t_normalized = t.float() / float(self.num_timesteps)
        return self.unet(x, t_normalized)

class D3PM(nn.Module):
    def __init__(self, vocab_size=256, hidden_dim=128, num_timesteps=1000):
        super().__init__()
        print("INSIDE D3PM: ", vocab_size, hidden_dim, num_timesteps)
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_timesteps = num_timesteps
        
        self.pixel_embedding = nn.Embedding(vocab_size, hidden_dim)
        
        time_emb_dim = hidden_dim * 4
        self.time_embedding = TimestepEmbedding(time_emb_dim)
        
        self.init_conv = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        
        self.down1 = ResidualBlock(hidden_dim, hidden_dim, time_emb_dim)
        self.down2 = ResidualBlock(hidden_dim, hidden_dim * 2, time_emb_dim)
        self.down3 = ResidualBlock(hidden_dim * 2, hidden_dim * 4, time_emb_dim)
        
        self.bottleneck = ResidualBlock(hidden_dim * 4, hidden_dim * 4, time_emb_dim)
        
        self.up3 = ResidualBlock(hidden_dim * 8, hidden_dim * 2, time_emb_dim)
        self.up2 = ResidualBlock(hidden_dim * 4, hidden_dim, time_emb_dim)
        self.up1 = ResidualBlock(hidden_dim * 2, hidden_dim, time_emb_dim)
        
        self.classifier = nn.Conv2d(hidden_dim, vocab_size, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
    def forward(self, x, t):
        x_clamped = torch.clamp(x.long(), 0, self.vocab_size - 1)
        
        x_emb = self.pixel_embedding(x_clamped)
        x_emb = x_emb.permute(0, 3, 1, 2)
        
        time_emb = self.time_embedding(t)
        
        x = self.init_conv(x_emb)
        
        skip1 = self.down1(x, time_emb)
        x = self.pool(skip1)
        
        skip2 = self.down2(x, time_emb)
        x = self.pool(skip2)
        
        skip3 = self.down3(x, time_emb)
        x = self.pool(skip3)
        
        x = self.bottleneck(x, time_emb)
        
        x = self.upsample(x)
        if x.shape[-2:] != skip3.shape[-2:]:
            skip3 = F.interpolate(skip3, size=x.shape[-2:], mode='nearest')
        x = torch.cat([x, skip3], dim=1)
        x = self.up3(x, time_emb)
        
        x = self.upsample(x)
        if x.shape[-2:] != skip2.shape[-2:]:
            skip2 = F.interpolate(skip2, size=x.shape[-2:], mode='nearest')
        x = torch.cat([x, skip2], dim=1)
        x = self.up2(x, time_emb)
        
        x = self.upsample(x)
        if x.shape[-2:] != skip1.shape[-2:]:
            skip1 = F.interpolate(skip1, size=x.shape[-2:], mode='nearest')
        x = torch.cat([x, skip1], dim=1)
        x = self.up1(x, time_emb)
        
        if x.shape[-2:] != (28, 28):
            x = F.interpolate(x, size=(28, 28), mode='nearest')
        
        logits = self.classifier(x)
        
        return logits

class ConditionalDDPM(nn.Module):
    def __init__(self, num_classes=10, num_timesteps=1000):
        super().__init__()
        self.num_classes = num_classes
        self.unet = UNet(in_channels=1, out_channels=1, time_dim=128, num_classes=num_classes)
        self.num_timesteps = num_timesteps
    
    def forward(self, x, t, y):
        t_normalized = t.float() / float(self.num_timesteps)
        return self.unet(x, t_normalized, y)
        
class ConditionalD3PM(nn.Module):
    def __init__(self, vocab_size=256, hidden_dim=128, num_timesteps=1000, num_classes=10):
        super().__init__()
        print("INSIDE CONDITIONAL D3PM: ", vocab_size, hidden_dim, num_timesteps)
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_timesteps = num_timesteps
        self.num_classes = num_classes
        
        self.pixel_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.class_embedding = nn.Embedding(num_classes, hidden_dim)
        
        time_emb_dim = hidden_dim * 4
        self.time_embedding = TimestepEmbedding(time_emb_dim)
        
        self.combined_embedding = nn.Linear(time_emb_dim + hidden_dim, time_emb_dim)
        
        self.init_conv = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        
        self.down1 = ResidualBlock(hidden_dim, hidden_dim, time_emb_dim)
        self.down2 = ResidualBlock(hidden_dim, hidden_dim * 2, time_emb_dim)
        self.down3 = ResidualBlock(hidden_dim * 2, hidden_dim * 4, time_emb_dim)
        
        self.bottleneck = ResidualBlock(hidden_dim * 4, hidden_dim * 4, time_emb_dim)
        
        self.up3 = ResidualBlock(hidden_dim * 8, hidden_dim * 2, time_emb_dim)
        self.up2 = ResidualBlock(hidden_dim * 4, hidden_dim, time_emb_dim)
        self.up1 = ResidualBlock(hidden_dim * 2, hidden_dim, time_emb_dim)
        
        self.classifier = nn.Conv2d(hidden_dim, vocab_size, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
    def forward(self, x, t, class_labels):
        x_clamped = torch.clamp(x.long(), 0, self.vocab_size - 1)
        
        x_emb = self.pixel_embedding(x_clamped)
        x_emb = x_emb.permute(0, 3, 1, 2)
        
        time_emb = self.time_embedding(t)
        class_emb = self.class_embedding(class_labels)
        
        combined_emb = torch.cat([time_emb, class_emb], dim=-1)
        combined_emb = self.combined_embedding(combined_emb)
        combined_emb = nn.ReLU()(combined_emb)
        
        x = self.init_conv(x_emb)
        
        skip1 = self.down1(x, combined_emb)
        x = self.pool(skip1)
        
        skip2 = self.down2(x, combined_emb)
        x = self.pool(skip2)
        
        skip3 = self.down3(x, combined_emb)
        x = self.pool(skip3)
        
        x = self.bottleneck(x, combined_emb)
        
        x = self.upsample(x)
        if x.shape[-2:] != skip3.shape[-2:]:
            skip3 = F.interpolate(skip3, size=x.shape[-2:], mode='nearest')
        x = torch.cat([x, skip3], dim=1)
        x = self.up3(x, combined_emb)
        
        x = self.upsample(x)
        if x.shape[-2:] != skip2.shape[-2:]:
            skip2 = F.interpolate(skip2, size=x.shape[-2:], mode='nearest')
        x = torch.cat([x, skip2], dim=1)
        x = self.up2(x, combined_emb)
        
        x = self.upsample(x)
        if x.shape[-2:] != skip1.shape[-2:]:
            skip1 = F.interpolate(skip1, size=x.shape[-2:], mode='nearest')
        x = torch.cat([x, skip1], dim=1)
        x = self.up1(x, combined_emb)
        
        if x.shape[-2:] != (28, 28):
            x = F.interpolate(x, size=(28, 28), mode='nearest')
        
        logits = self.classifier(x)
        
        return logits