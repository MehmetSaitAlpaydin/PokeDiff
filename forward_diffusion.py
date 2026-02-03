"""
Forward Diffusion Process
Demonstrates how to add noise to images gradually over time.

This is the EASY part - just add Gaussian noise according to a schedule.
The hard part (reverse process) comes later.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image


class NoiseScheduler:
    """
    Defines how much noise to add at each timestep.
    
    Uses linear schedule: gradually increase noise from β_start to β_end
    """
    
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps
        
        # β schedule: how much noise to add at each step
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        
        # α = 1 - β: how much of original signal to keep
        self.alphas = 1.0 - self.betas
        
        # ᾱ = cumulative product of alphas
        # This lets us jump to any timestep directly!
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # ᾱ_(t-1) for the reverse process
        self.alphas_cumprod_prev = torch.cat([
            torch.tensor([1.0]), 
            self.alphas_cumprod[:-1]
        ])
        
        # Pre-compute useful terms for sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
    def add_noise(self, x_0, t, noise=None):
        """
        Add noise to image x_0 at timestep t.
        
        Formula: x_t = √(ᾱ_t) * x_0 + √(1 - ᾱ_t) * ε
        
        Args:
            x_0: Original image (batch, channels, height, width)
            t: Timestep(s) - can be single int or tensor of ints
            noise: Optional pre-generated noise (for reproducibility)
        
        Returns:
            x_t: Noisy image at timestep t
            noise: The noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # Ensure t is long type for indexing
        if isinstance(t, torch.Tensor):
            t = t.long()
        
        # Get coefficients for this timestep
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t]
        
        # Reshape for broadcasting with 4D images (batch, channels, height, width)
        if len(x_0.shape) == 4 and sqrt_alpha_cumprod.ndim == 1:
            sqrt_alpha_cumprod = sqrt_alpha_cumprod[:, None, None, None]
            sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod[:, None, None, None]
        
        # Apply noise formula
        x_t = sqrt_alpha_cumprod * x_0 + sqrt_one_minus_alpha_cumprod * noise
        
        return x_t, noise


def demonstrate_forward_process():
    """
    Visualize how an image becomes progressively noisier.
    This demonstrates the forward diffusion process.
    """
    print("Demonstrating Forward Diffusion Process")
    print("=" * 60)
    
    # Create noise scheduler
    scheduler = NoiseScheduler(num_timesteps=1000)
    
    # Load a sample image (or create a simple one)
    # For demo, let's create a simple gradient image
    print("\nCreating sample image...")
    img = torch.zeros(1, 3, 64, 64)
    # Create a circular gradient pattern (looks like a simple shape)
    for i in range(64):
        for j in range(64):
            dist = np.sqrt((i - 32)**2 + (j - 32)**2)
            img[0, :, i, j] = 1.0 - min(dist / 32, 1.0)
    
    # Normalize to [-1, 1] range (standard for diffusion models)
    img = img * 2 - 1
    
    # Show progressive noising
    timesteps_to_show = [0, 50, 100, 250, 500, 750, 999]
    
    print(f"\nAdding noise at different timesteps:")
    print(f"Timesteps to visualize: {timesteps_to_show}")
    print()
    
    fig, axes = plt.subplots(1, len(timesteps_to_show), figsize=(15, 3))
    
    for idx, t in enumerate(timesteps_to_show):
        if t == 0:
            # Original image
            noisy_img = img
        else:
            # Add noise at timestep t
            t_tensor = torch.tensor([t])
            noisy_img, _ = scheduler.add_noise(img, t_tensor)
        
        # Convert to viewable format [0, 1]
        display_img = (noisy_img[0] + 1) / 2  # [-1, 1] → [0, 1]
        display_img = torch.clamp(display_img, 0, 1)
        
        # Plot
        axes[idx].imshow(display_img.permute(1, 2, 0).numpy())
        axes[idx].set_title(f't={t}')
        axes[idx].axis('off')
        
        # Calculate signal-to-noise ratio
        if t > 0:
            alpha_cumprod = scheduler.alphas_cumprod[t].item()
            signal_ratio = alpha_cumprod * 100
            noise_ratio = (1 - alpha_cumprod) * 100
            print(f"t={t:3d}: Signal={signal_ratio:5.1f}%, Noise={noise_ratio:5.1f}%")
    
    plt.tight_layout()
    plt.savefig('forward_diffusion_demo.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: forward_diffusion_demo.png")
    print()
    
    # Show noise schedule
    print("Noise Schedule Visualization:")
    print("-" * 60)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot beta schedule
    ax1.plot(scheduler.betas.numpy())
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('β_t (noise added per step)')
    ax1.set_title('Noise Schedule (β)')
    ax1.grid(True, alpha=0.3)
    
    # Plot cumulative alpha
    ax2.plot(scheduler.alphas_cumprod.numpy())
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('ᾱ_t (cumulative signal retention)')
    ax2.set_title('Cumulative Signal Retention (ᾱ)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('noise_schedule.png', dpi=150, bbox_inches='tight')
    print(f"Saved noise schedule to: noise_schedule.png")
    print()
    
    # Key properties
    print("Key Properties:")
    print("-" * 60)
    print(f"Total timesteps: {scheduler.num_timesteps}")
    print(f"β_start: {scheduler.betas[0]:.6f}")
    print(f"β_end: {scheduler.betas[-1]:.6f}")
    print(f"At t=500, ᾱ={scheduler.alphas_cumprod[500]:.4f}")
    print(f"  → Image is {scheduler.alphas_cumprod[500]*100:.1f}% original")
    print(f"  → Image is {(1-scheduler.alphas_cumprod[500])*100:.1f}% noise")
    print(f"At t=999, ᾱ={scheduler.alphas_cumprod[999]:.6f}")
    print(f"  → Almost pure noise!")
    print()
    print("✓ Forward diffusion complete!")
    print("  The REVERSE process (removing noise) is what we'll train a neural network to do.")


if __name__ == '__main__':
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    demonstrate_forward_process()
