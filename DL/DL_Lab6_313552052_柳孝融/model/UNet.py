import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for timestep encoding"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)                               # determines the range of frequencies 0~embeddings
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings) # Creates a tensor of values [1, e^(-k), e^(-2k), ..., e^(-127k)] where k is the value from line 1
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ConvBlock(nn.Module):
    """Basic convolutional block for U-Net"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class DownBlock(nn.Module):
    """Downsampling block for U-Net"""
    def __init__(self, in_channels, out_channels, emb_dim):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.emb_layer = nn.Linear(emb_dim, out_channels)
        self.downsample = nn.MaxPool2d(2)
        
    def forward(self, x, t_emb):
        # Apply convolution
        x = self.conv(x)
        # Add time embedding
        emb = self.emb_layer(t_emb)[:, :, None, None].repeat(1, 1, x.shape[2], x.shape[3])
        x = x + emb
        # Downsample
        return x, self.downsample(x)

class UpBlock(nn.Module):
    """Upsampling block for U-Net"""
    def __init__(self, in_channels, out_channels, emb_dim):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels + in_channels, out_channels)
        self.emb_layer = nn.Linear(emb_dim, out_channels)
        
    def forward(self, x, skip, t_emb):
        # Upsample
        x = self.upsample(x)
        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        # Apply convolution
        x = self.conv(x)
        # Add time embedding
        emb = self.emb_layer(t_emb)[:, :, None, None].repeat(1, 1, x.shape[2], x.shape[3])
        x = x + emb
        return x

class SelfAttention(nn.Module):
    """Self-attention block for capturing long-range dependencies"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True) # 4 heads 
        self.ln = nn.LayerNorm([channels])                              # according to channel
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        
    def forward(self, x):
        size = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x                                   # residual connection
        attention_value = self.ff_self(attention_value) + attention_value       # feed-forward self-attention
        return attention_value.transpose(1, 2).reshape(-1, self.channels, *size)# unpack height and width dimensions 

class UNet(nn.Module):
    """Conditional U-Net model for diffusion"""
    def __init__(self, in_channels=3, out_channels=3, time_dim=256, condition_dim=24):
        super().__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),              # use the cumulative distribution function for gaussian distribution, smoother than ReLU
            nn.Linear(time_dim, time_dim),
        )
        
        # Condition embedding
        self.condition_mlp = nn.Sequential(
            nn.Linear(condition_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        
        # Downsampling path
        self.down1 = DownBlock(64, 128, time_dim)
        self.down2 = DownBlock(128, 256, time_dim)
        self.down3 = DownBlock(256, 512, time_dim)
        
        # Bottleneck with attention
        self.bottleneck = nn.Sequential(
            ConvBlock(512, 512),
            SelfAttention(512),
            ConvBlock(512, 512)
        )
        
        # Upsampling path
        self.up1 = UpBlock(512, 256, time_dim)
        self.up2 = UpBlock(256, 128, time_dim)
        self.up3 = UpBlock(128, 64, time_dim)
        
        # Final convolution
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x, t, condition):
        # Get time embedding
        t_emb = self.time_mlp(t)
        
        # Get condition embedding
        c_emb = self.condition_mlp(condition)
        
        # Combine time and condition embeddings
        emb = t_emb + c_emb
        
        # Initial convolution
        x = self.init_conv(x)
        
        # Downsampling
        skip1, x = self.down1(x, emb)
        skip2, x = self.down2(x, emb)
        skip3, x = self.down3(x, emb)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Upsampling
        x = self.up1(x, skip3, emb)
        x = self.up2(x, skip2, emb)
        x = self.up3(x, skip1, emb)
        
        # Final convolution
        x = self.final_conv(x)
        
        return x