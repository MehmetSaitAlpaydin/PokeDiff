# PokeDiff 🎮✨

**A Denoising Diffusion Probabilistic Model (DDPM) for generating Pokemon images from scratch.**

Built from the ground up to understand diffusion models - no pre-trained models, no black boxes, just pure implementation and learning.

---

## 🎯 Project Overview

PokeDiff implements a complete diffusion model pipeline:
- **Forward Diffusion**: Gradually adds Gaussian noise to images over 1000 timesteps
- **U-Net Architecture**: 54M parameter neural network with time embeddings and skip connections
- **Reverse Diffusion**: Trained model denoises random noise into Pokemon images
- **Data Augmentation**: Random flips, rotations, and color jitter to prevent overfitting

## 🏗️ Architecture

### Forward Process
```
Clean Pokemon → Add Noise (1000 steps) → Pure Noise
x_t = √(ᾱ_t) * x_0 + √(1 - ᾱ_t) * ε
```

### U-Net Model
- **Encoder**: 4 downsampling blocks (64 → 128 → 256 → 512 channels)
- **Decoder**: 4 upsampling blocks with skip connections
- **Time Embeddings**: Sinusoidal position embeddings for timestep conditioning
- **Activation**: SiLU (Swish)
- **Normalization**: GroupNorm

### Training
- Simple supervised learning: predict the noise that was added
- **Loss**: MSE between predicted noise and actual noise
- Much more stable than GANs (no adversarial training)

## 📁 Project Structure

```
Pokediff/
├── README.md                 # This file
├── DIFFUSION_THEORY.md      # Complete theory and math explanation
├── forward_diffusion.py     # Forward process and noise scheduler
├── unet.py                  # U-Net architecture
├── train.py                 # Training script
├── generate.py              # Generate new Pokemon from trained model
├── requirements.txt         # Python dependencies
├── data/                    # Pokemon dataset (819 images)
│   └── pokemon_jpg/
├── outputs/                 # Generated samples during training
└── checkpoints/            # Saved model weights
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python train.py
```

Training hyperparameters:
- **Epochs**: 2000
- **Batch Size**: 16
- **Learning Rate**: 0.0001
- **Image Size**: 64×64
- **Timesteps**: 1000

### 3. Generate Pokemon
```bash
python generate.py --checkpoint checkpoints/final_model.pt --num_images 64
```

## 📊 Training Progress

The model learns progressively:
- **Epoch 100**: Basic colors and shapes
- **Epoch 300**: Recognizable Pokemon structures
- **Epoch 500**: Good quality with diversity
- **Epoch 1000-2000**: High-quality, varied Pokemon

Generated samples are saved every 10 epochs to `outputs/generated_epoch_XXXX.png`.

## 🧠 Key Learnings

### Why Diffusion > GAN?
1. **Stability**: Simple MSE loss vs. adversarial training
2. **Quality**: 1000 small denoising steps vs. 1 big generation step
3. **Diversity**: No mode collapse issues
4. **Training**: Easier to debug and converge

### Implementation Details
- **Data Augmentation**: Essential with only 819 images
  - Random horizontal flip (50%)
  - Random rotation (±15°)
  - Color jitter (brightness, contrast, saturation, hue)
- **Noise Schedule**: Linear β from 0.0001 to 0.02
- **Optimizer**: Adam with learning rate 0.0001
- **Device**: GPU (CUDA) recommended, ~24GB VRAM for batch size 16

## 📈 Results

Training on 819 Pokemon images with data augmentation:
- **Model Size**: 54,195,523 parameters
- **Training Time**: ~X hours on RTX 4090 (2000 epochs)
- **Quality**: High-quality diverse Pokemon generation
- **Diversity**: Successfully avoids memorization through augmentation

## 🛠️ Technical Implementation

### Forward Diffusion
Implements the noise addition process:
- Pre-computes αt and ᾱt for efficiency
- Allows jumping to any timestep directly
- Handles batched operations for GPU acceleration

### U-Net Architecture
- Custom implementation with time conditioning
- Skip connections between encoder/decoder
- GroupNorm for stable training
- Sinusoidal time embeddings (like Transformers)

### Training Loop
```python
for epoch in range(num_epochs):
    for images in dataloader:
        # Sample random timestep
        t = random_timestep()
        # Add noise
        noisy_image = add_noise(image, t)
        # Predict noise
        predicted_noise = model(noisy_image, t)
        # Calculate loss
        loss = MSE(predicted_noise, actual_noise)
        # Backpropagate
        loss.backward()
```

## 🎓 Theory

See [DIFFUSION_THEORY.md](DIFFUSION_THEORY.md) for:
- Complete mathematical derivation
- Intuitive analogies (ink drop in water)
- Comparison with GANs
- Architecture design choices
- Training strategy explanation

## 🔮 Future Improvements

- [ ] Conditional generation (control Pokemon type/attributes)
- [ ] DDIM sampling (faster generation - 50 steps instead of 1000)
- [ ] Higher resolution (128×128 or 256×256)
- [ ] Latent diffusion (compress to latent space)
- [ ] Classifier-free guidance (better quality)

## 📚 References

- [Denoising Diffusion Probabilistic Models (DDPM)](https://arxiv.org/abs/2006.11239)
- [Understanding Diffusion Models: A Unified Perspective](https://arxiv.org/abs/2208.11970)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

## 🙏 Acknowledgments

- Pokemon dataset from Kaggle
- Built as part of learning journey to understand DeepFaceLive and modern generative models
- No pre-trained models used - everything implemented from scratch for learning

## 📝 License

MIT License - Feel free to use for learning and experimentation!

---

**Built with PyTorch 🔥 | Trained on NVIDIA RTX 4090 💪 | Generated with ❤️**
