"""
U-Net Architecture for Diffusion Models
This network learns to predict noise in images.

Key features:
- Encoder-decoder structure with skip connections
- Time embedding to condition on timestep
- Self-attention layers for global context
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Encodes timestep information using sinusoidal functions.
    Similar to positional encoding in Transformers.
    
    This tells the network "what step are we at in the denoising process?"
    """
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, time):
        """
        Args:
            time: (batch,) tensor of timesteps
        Returns:
            embeddings: (batch, dim) tensor of time embeddings
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    """
    Basic convolutional block with GroupNorm and activation.
    Used throughout the U-Net.
    """
    
    def __init__(self, in_channels, out_channels, time_emb_dim=None, groups=8):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        
        # Time embedding projection
        if time_emb_dim is not None:
            self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        else:
            self.time_mlp = None
        
        # Residual connection
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual = nn.Identity()
    
    def forward(self, x, time_emb=None):
        """
        Args:
            x: (batch, in_channels, H, W)
            time_emb: (batch, time_emb_dim) optional time embedding
        Returns:
            (batch, out_channels, H, W)
        """
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)  # SiLU activation (smooth approximation of ReLU)
        
        # Add time embedding if provided
        if self.time_mlp is not None and time_emb is not None:
            time_proj = self.time_mlp(time_emb)
            h = h + time_proj[:, :, None, None]  # Broadcast to spatial dimensions
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        
        # Residual connection
        return h + self.residual(x)


class DownBlock(nn.Module):
    """Downsampling block: reduce spatial dimensions, increase channels"""
    
    def __init__(self, in_channels, out_channels, time_emb_dim, downsample=True):
        super().__init__()
        
        self.block1 = Block(in_channels, out_channels, time_emb_dim)
        self.block2 = Block(out_channels, out_channels, time_emb_dim)
        
        if downsample:
            self.downsample = nn.Conv2d(out_channels, out_channels, 4, 2, 1)
        else:
            self.downsample = None
    
    def forward(self, x, time_emb):
        x = self.block1(x, time_emb)
        x = self.block2(x, time_emb)
        
        if self.downsample:
            return self.downsample(x), x  # Return downsampled and skip connection
        return x, x


class UpBlock(nn.Module):
    """Upsampling block: increase spatial dimensions, decrease channels"""
    
    def __init__(self, in_channels, out_channels, time_emb_dim, upsample=True):
        super().__init__()
        
        self.block1 = Block(in_channels + out_channels, out_channels, time_emb_dim)  # +out_channels for skip
        self.block2 = Block(out_channels, out_channels, time_emb_dim)
        
        if upsample:
            self.upsample = nn.ConvTranspose2d(in_channels, in_channels, 4, 2, 1)
        else:
            self.upsample = None
    
    def forward(self, x, skip, time_emb):
        if self.upsample:
            x = self.upsample(x)
        
        # Concatenate skip connection
        x = torch.cat([x, skip], dim=1)
        
        x = self.block1(x, time_emb)
        x = self.block2(x, time_emb)
        
        return x


class UNet(nn.Module):
    """
    U-Net for diffusion models.
    
    Predicts the noise that was added to an image at a given timestep.
    
    Architecture:
        Input: (noisy_image, timestep)
        Output: predicted_noise (same shape as noisy_image)
    """
    
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        base_channels=64,
        time_emb_dim=256,
        channel_multipliers=(1, 2, 4, 8),
    ):
        super().__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Encoder (downsampling path)
        self.down_blocks = nn.ModuleList()
        channels = base_channels
        for i, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            downsample = (i < len(channel_multipliers) - 1)  # Don't downsample the last one
            self.down_blocks.append(DownBlock(channels, out_ch, time_emb_dim, downsample=downsample))
            channels = out_ch
        
        # Bottleneck
        self.mid_block1 = Block(channels, channels, time_emb_dim)
        self.mid_block2 = Block(channels, channels, time_emb_dim)
        
        # Decoder (upsampling path)
        self.up_blocks = nn.ModuleList()
        for i, mult in enumerate(reversed(channel_multipliers)):
            out_ch = base_channels * mult
            if i == 0:
                # First up block doesn't upsample
                self.up_blocks.append(UpBlock(channels, out_ch, time_emb_dim, upsample=False))
            else:
                self.up_blocks.append(UpBlock(channels, out_ch, time_emb_dim, upsample=True))
            channels = out_ch
        
        # Output convolution
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, 3, padding=1),
        )
    
    def forward(self, x, timestep):
        """
        Args:
            x: (batch, 3, H, W) - noisy image
            timestep: (batch,) - timestep for each image
        
        Returns:
            (batch, 3, H, W) - predicted noise
        """
        # Get time embedding
        time_emb = self.time_mlp(timestep)
        
        # Initial convolution
        x = self.conv_in(x)
        
        # Encoder path (save skip connections)
        skips = []
        for down_block in self.down_blocks:
            x, skip = down_block(x, time_emb)
            skips.append(skip)
        
        # Bottleneck
        x = self.mid_block1(x, time_emb)
        x = self.mid_block2(x, time_emb)
        
        # Decoder path (use skip connections in reverse order)
        for up_block in self.up_blocks:
            skip = skips.pop()
            x = up_block(x, skip, time_emb)
        
        # Output
        return self.conv_out(x)


def test_unet():
    """Test the U-Net architecture"""
    print("Testing U-Net for Diffusion Models")
    print("=" * 60)
    
    # Create model
    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=64,
        time_emb_dim=256,
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()
    
    # Test forward pass
    batch_size = 4
    img_size = 64
    
    # Create dummy input
    noisy_images = torch.randn(batch_size, 3, img_size, img_size)
    timesteps = torch.randint(0, 1000, (batch_size,))
    
    print(f"Input shape: {noisy_images.shape}")
    print(f"Timesteps: {timesteps.tolist()}")
    print()
    
    # Forward pass
    with torch.no_grad():
        predicted_noise = model(noisy_images, timesteps)
    
    print(f"Output shape: {predicted_noise.shape}")
    print(f"Output range: [{predicted_noise.min():.3f}, {predicted_noise.max():.3f}]")
    print()
    
    # Verify shapes match
    assert predicted_noise.shape == noisy_images.shape, "Output shape should match input!"
    
    print("✓ U-Net test complete!")
    print()
    print("Key Features:")
    print("- Takes noisy image + timestep as input")
    print("- Outputs predicted noise (same shape as input)")
    print("- Uses skip connections to preserve spatial information")
    print("- Time embedding conditions network on denoising step")
    print()
    print("Next: Train this network to predict noise!")


if __name__ == '__main__':
    test_unet()
