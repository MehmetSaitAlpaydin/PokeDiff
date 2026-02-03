"""
Quick test to verify PokeDiff setup is correct.
Run this before starting training to catch any issues early.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all required packages are installed"""
    print("Testing imports...")
    try:
        import torch
        import torchvision
        import PIL
        import tqdm
        import matplotlib
        print("✓ All packages installed correctly")
        return True
    except ImportError as e:
        print(f"✗ Missing package: {e}")
        print("Run: pip install -r requirements.txt")
        return False

def test_cuda():
    """Test CUDA availability"""
    print("\nTesting CUDA...")
    import torch
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        return True
    else:
        print("⚠ CUDA not available - training will be slow on CPU")
        return False

def test_dataset():
    """Test that dataset exists and is readable"""
    print("\nTesting dataset...")
    data_dir = Path('data/pokemon_jpg')
    if not data_dir.exists():
        print(f"✗ Dataset directory not found: {data_dir}")
        return False
    
    images = list(data_dir.glob('*.jpg'))
    if len(images) == 0:
        print(f"✗ No images found in {data_dir}")
        return False
    
    print(f"✓ Found {len(images)} Pokemon images")
    return True

def test_modules():
    """Test that project modules can be imported"""
    print("\nTesting project modules...")
    try:
        from unet import UNet
        from forward_diffusion import NoiseScheduler
        print("✓ All project modules import correctly")
        
        # Test instantiation
        print("\nTesting model instantiation...")
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        model = UNet(
            in_channels=3,
            out_channels=3,
            base_channels=64,
            time_emb_dim=256,
        ).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model created: {total_params:,} parameters")
        
        scheduler = NoiseScheduler(num_timesteps=1000, beta_start=0.0001, beta_end=0.02)
        print(f"✓ Scheduler created: {scheduler.num_timesteps} timesteps")
        
        return True
    except Exception as e:
        print(f"✗ Error importing modules: {e}")
        return False

def test_forward_pass():
    """Test a forward pass through the model"""
    print("\nTesting forward pass...")
    try:
        import torch
        from unet import UNet
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = UNet(in_channels=3, out_channels=3, base_channels=64, time_emb_dim=256).to(device)
        model.eval()
        
        # Create dummy input
        batch_size = 2
        x = torch.randn(batch_size, 3, 64, 64, device=device)
        t = torch.randint(0, 1000, (batch_size,), device=device).long()
        
        # Forward pass
        with torch.no_grad():
            output = model(x, t)
        
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_directories():
    """Test that output directories exist"""
    print("\nTesting directories...")
    dirs = ['outputs', 'checkpoints']
    all_good = True
    for d in dirs:
        path = Path(d)
        if path.exists():
            print(f"✓ {d}/ exists")
        else:
            print(f"⚠ {d}/ doesn't exist (will be created)")
            path.mkdir(exist_ok=True)
            print(f"  Created {d}/")
    return all_good

def main():
    print("=" * 60)
    print("PokeDiff Setup Verification")
    print("=" * 60)
    print()
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("CUDA", test_cuda()))
    results.append(("Dataset", test_dataset()))
    results.append(("Modules", test_modules()))
    results.append(("Forward Pass", test_forward_pass()))
    results.append(("Directories", test_directories()))
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:.<30} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n🎉 All tests passed! Ready to train.")
        print("\nNext step: python train.py")
    else:
        print("\n⚠ Some tests failed. Please fix issues before training.")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
