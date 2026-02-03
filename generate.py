"""
PokeDiff - Generate Pokemon
Generate new Pokemon images from a trained model.
"""

import torch
from torchvision.utils import save_image
from pathlib import Path
import argparse
from tqdm import tqdm

from unet import UNet
from forward_diffusion import NoiseScheduler


def generate_pokemon(
    checkpoint_path,
    num_images=64,
    output_path='generated.png',
    image_size=64,
    device='cuda'
):
    """
    Generate Pokemon images from trained model.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        num_images: Number of Pokemon to generate
        output_path: Where to save generated images
        image_size: Size of generated images (64x64)
        device: 'cuda' or 'cpu'
    """
    
    # Load model
    print(f"Loading model from {checkpoint_path}")
    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=64,
        time_emb_dim=256,
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print(f"Model loaded successfully!")
    
    # Create scheduler
    scheduler = NoiseScheduler(
        num_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02
    )
    scheduler.betas = scheduler.betas.to(device)
    scheduler.alphas = scheduler.alphas.to(device)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
    
    # Generate
    print(f"Generating {num_images} Pokemon...")
    with torch.no_grad():
        # Start with pure noise
        x = torch.randn(num_images, 3, image_size, image_size, device=device)
        
        # Gradually denoise
        for t in tqdm(reversed(range(scheduler.num_timesteps)), desc="Sampling"):
            # Create timestep tensor
            t_batch = torch.full((num_images,), t, device=device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = model(x, t_batch)
            
            # Get coefficients
            alpha = scheduler.alphas[t]
            alpha_cumprod = scheduler.alphas_cumprod[t]
            beta = scheduler.betas[t]
            
            # Remove predicted noise
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            
            x = (1 / torch.sqrt(alpha)) * (
                x - (beta / torch.sqrt(1 - alpha_cumprod)) * predicted_noise
            )
            
            if t > 0:
                x = x + torch.sqrt(beta) * noise
    
    # Denormalize from [-1, 1] to [0, 1]
    x = (x + 1) / 2
    x = torch.clamp(x, 0, 1)
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_image(x, output_path, nrow=8)
    print(f"\n✓ Generated {num_images} Pokemon!")
    print(f"✓ Saved to: {output_path}")
    
    return x


def main():
    parser = argparse.ArgumentParser(description='Generate Pokemon with PokeDiff')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--num_images',
        type=int,
        default=64,
        help='Number of Pokemon to generate (default: 64)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='generated_pokemon.png',
        help='Output file path (default: generated_pokemon.png)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (default: cuda if available)'
    )
    
    args = parser.parse_args()
    
    # Check checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        print("\nAvailable checkpoints:")
        checkpoint_dir = Path('checkpoints')
        if checkpoint_dir.exists():
            for cp in checkpoint_dir.glob('*.pt'):
                print(f"  - {cp}")
        return
    
    # Generate
    generate_pokemon(
        checkpoint_path=args.checkpoint,
        num_images=args.num_images,
        output_path=args.output,
        device=args.device
    )


if __name__ == '__main__':
    main()
