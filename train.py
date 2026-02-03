"""
PokeDiff - Pokemon Diffusion Model
Train a diffusion model to generate Pokemon images from scratch.

Training is MUCH simpler than GAN:
1. Sample random image from dataset
2. Sample random timestep
3. Add noise to image at that timestep
4. Predict the noise with U-Net
5. Calculate MSE loss
6. Backpropagate

No adversarial training, no discriminator, just simple supervised learning!
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from forward_diffusion import NoiseScheduler
from unet import UNet


# Hyperparameters
BATCH_SIZE = 16  # Smaller than GAN due to memory (1000 timesteps!)
LEARNING_RATE = 0.0001
NUM_EPOCHS = 5000  # Long training for high-quality results
IMAGE_SIZE = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Diffusion parameters
NUM_TIMESTEPS = 1000
BETA_START = 0.0001
BETA_END = 0.02

# Directories
DATA_DIR = Path('data/pokemon_jpg')
OUTPUT_DIR = Path('outputs')
OUTPUT_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR = Path('checkpoints')
CHECKPOINT_DIR.mkdir(exist_ok=True)


class PokemonDataset(Dataset):
    """Dataset of Pokémon images with augmentation"""
    def __init__(self, data_dir, transform=None, augment=True):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.augment = augment
        self.image_files = list(self.data_dir.glob('*.jpg'))
        print(f"Found {len(self.image_files)} images")
        
        # Augmentation transforms (applied randomly during training)
        if self.augment:
            self.aug_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Apply augmentation BEFORE normalize (augment on 0-255 images)
        if self.augment:
            image = self.aug_transform(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image


@torch.no_grad()
def sample_images(model, scheduler, num_images=64, device='cuda'):
    """
    Generate images using the trained model.
    
    This is the REVERSE process: start with noise, gradually denoise.
    """
    model.eval()
    
    # Start with pure noise
    x = torch.randn(num_images, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    
    # Gradually denoise (from T to 0)
    for t in reversed(range(scheduler.num_timesteps)):
        # Create timestep tensor for all images
        t_batch = torch.full((num_images,), t, device=device, dtype=torch.long)
        
        # Predict noise
        predicted_noise = model(x, t_batch)
        
        # Get coefficients
        alpha = scheduler.alphas[t]
        alpha_cumprod = scheduler.alphas_cumprod[t]
        beta = scheduler.betas[t]
        
        # Remove predicted noise
        # x_(t-1) = 1/√α_t * (x_t - (β_t / √(1-ᾱ_t)) * predicted_noise)
        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)
        
        x = (1 / torch.sqrt(alpha)) * (
            x - (beta / torch.sqrt(1 - alpha_cumprod)) * predicted_noise
        )
        
        if t > 0:
            # Add noise back (except at last step)
            x = x + torch.sqrt(beta) * noise
    
    model.train()
    return x


def train():
    """Train the diffusion model"""
    print("=" * 80)
    print("PokeDiff - Pokemon Diffusion Model Training")
    print("=" * 80)
    print()
    print(f"Device: {DEVICE}")
    
    if DEVICE.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    print("=" * 80)
    print()
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1, 1]
    ])
    
    # Load dataset
    dataset = PokemonDataset(DATA_DIR, transform=transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0,  # Avoid Windows multiprocessing issues
        pin_memory=True
    )
    print(f"Loaded {len(dataset)} images")
    print(f"Batches per epoch: {len(dataloader)}")
    print()
    
    # Create model and noise scheduler
    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=64,
        time_emb_dim=256,
    ).to(DEVICE)
    
    scheduler = NoiseScheduler(
        num_timesteps=NUM_TIMESTEPS,
        beta_start=BETA_START,
        beta_end=BETA_END
    )
    # Move scheduler tensors to GPU
    scheduler.betas = scheduler.betas.to(DEVICE)
    scheduler.alphas = scheduler.alphas.to(DEVICE)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(DEVICE)
    scheduler.alphas_cumprod_prev = scheduler.alphas_cumprod_prev.to(DEVICE)
    scheduler.sqrt_alphas_cumprod = scheduler.sqrt_alphas_cumprod.to(DEVICE)
    scheduler.sqrt_one_minus_alphas_cumprod = scheduler.sqrt_one_minus_alphas_cumprod.to(DEVICE)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Loss function (simple MSE!)
    criterion = nn.MSELoss()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print()
    
    print("Starting training...")
    print(f"Timesteps: {NUM_TIMESTEPS}")
    print(f"Training is MUCH simpler than GAN - just predict noise!")
    print()
    
    # Training loop
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")
        for batch_idx, real_images in enumerate(pbar):
            real_images = real_images.to(DEVICE)
            batch_size = real_images.size(0)
            
            # Sample random timesteps for each image
            timesteps = torch.randint(
                0, scheduler.num_timesteps, 
                (batch_size,), 
                device=DEVICE
            ).long()
            
            # Add noise to images
            noise = torch.randn_like(real_images)
            noisy_images, _ = scheduler.add_noise(real_images, timesteps, noise)
            
            # Predict noise
            optimizer.zero_grad()
            predicted_noise = model(noisy_images, timesteps)
            
            # Calculate loss (just MSE between actual and predicted noise!)
            loss = criterion(predicted_noise, noise)
            
            # Backpropagate
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch}/{NUM_EPOCHS}] | Loss: {avg_loss:.4f}")
        
        # Checkpoints at strategic points: 1, 1000, 2000, 3000, 4000, 5000
        checkpoint_epochs = [1, 1000, 2000, 3000, 4000, 5000]
        
        # Generate sample images and save checkpoint at milestone epochs only
        if epoch in checkpoint_epochs:
            print(f"Generating samples...")
            with torch.no_grad():
                samples = sample_images(model, scheduler, num_images=64, device=DEVICE)
                # Denormalize from [-1, 1] to [0, 1]
                samples = (samples + 1) / 2
                samples = torch.clamp(samples, 0, 1)
                save_image(samples, OUTPUT_DIR / f'generated_epoch_{epoch:04d}.png', nrow=8)
            print(f"Saved generated images to {OUTPUT_DIR}/generated_epoch_{epoch:04d}.png")
            
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint, CHECKPOINT_DIR / f'checkpoint_epoch_{epoch}.pt')
            print(f"Saved checkpoint at epoch {epoch}")
        
        print()
    
    print("Training complete!")
    
    # Save final model
    final_checkpoint = {
        'model': model.state_dict(),
        'scheduler_config': {
            'num_timesteps': NUM_TIMESTEPS,
            'beta_start': BETA_START,
            'beta_end': BETA_END,
        }
    }
    torch.save(final_checkpoint, CHECKPOINT_DIR / 'final_model.pt')
    print(f"Saved final model to {CHECKPOINT_DIR}/final_model.pt")


if __name__ == '__main__':
    train()
