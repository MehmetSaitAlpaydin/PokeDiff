# Diffusion Models - Theory and Implementation

## The Core Concept

Diffusion models work by reversing a noise addition process. Consider how ink disperses in water:
- **T=0**: Clear ink drop (structured)
- **T=1**: Ink spreads slightly (less structured)
- **T=2**: Ink spreads more (even less structure)
- **T=1000**: Completely dispersed (pure noise)

Diffusion models learn to reverse this process.

**Forward Process** (noise addition):
```
Real Image → Slightly Noisy → More Noisy → ... → Pure Noise
```

**Reverse Process** (learned denoising):
```
Pure Noise → Less Noisy → Even Less Noisy → ... → Real Image
```

---

## Advantages Over GANs

### GAN Approach:
- Generator attempts to transform noise → image in one step
- Discriminator provides adversarial feedback
- Training can be unstable

### Diffusion Approach:
- Decomposes generation into 1000 small denoising steps
- Each step removes a small amount of noise
- Simpler learning objective leads to stable training

---

## Mathematical Framework

### Forward Diffusion Process

At each timestep t, Gaussian noise is added:

```
x_t = √(1 - β_t) * x_(t-1) + √β_t * ε

where:
  x_0 = original image
  x_t = noisy image at timestep t
  β_t = noise schedule (amount of noise to add)
  ε ~ N(0, 1) = random Gaussian noise
```

**Key property**: Direct sampling at any timestep:

```
x_t = √(ᾱ_t) * x_0 + √(1 - ᾱ_t) * ε

where ᾱ_t = product of (1 - β) up to t
```

This enables efficient training by sampling arbitrary noise levels.

### Reverse Process

A neural network ε_θ is trained to predict noise at each timestep:

```
ε_θ(x_t, t) → predicted_noise

Objective: predicted_noise ≈ actual_noise
Loss: MSE(predicted_noise, actual_noise)
```

**Generation procedure:**
1. Start with pure noise x_T ~ N(0, 1)
2. For t = T down to 1:
   - Predict noise: ε = ε_θ(x_t, t)
   - Remove noise: x_(t-1) = denoise(x_t, ε, t)
3. Output x_0 as generated image

---

## Architecture: U-Net

The denoising network uses a U-Net architecture:

```
Input: (noisy_image, timestep)
         ↓
    Encoder Path (downsampling)
    64 → 128 → 256 → 512 channels
         ↓
    Bottleneck
         ↓
    Decoder Path (upsampling)
    512 → 256 → 128 → 64 channels
    WITH skip connections ←---┐
         ↓                    |
    Output: predicted_noise   |
                              |
    Skip connections pass     |
    features from encoder ----┘
```

**U-Net Features:**
- Skip connections preserve spatial information
- Encoder extracts hierarchical features
- Decoder reconstructs noise map
- Time embedding conditions network on current timestep

---

## Noise Schedule

The noise addition rate is controlled by a schedule. Common implementations:

**Linear Schedule**:
```python
β_t = β_start + (β_end - β_start) * t/T
β_start = 0.0001
β_end = 0.02
T = 1000
```

**Cosine Schedule**:
```python
More gradual noise addition
Better behavior at extreme timesteps
```

---

## Training Algorithm

### Training Steps

1. **Sample image from dataset**
```python
x_0 = sample_from_dataset()
```

2. **Sample random timestep**
```python
t = random_uniform(1, 1000)
```

3. **Add noise to image**
```python
noise = random_normal()
x_t = √(ᾱ_t) * x_0 + √(1 - ᾱ_t) * noise
```

4. **Predict the noise**
```python
predicted_noise = model(x_t, t)
```

5. **Compute loss**
```python
loss = MSE(predicted_noise, noise)
```

6. **Update parameters**
```python
optimizer.step()
```

The training objective is straightforward noise prediction at random timesteps.

---

## Generation (Sampling)

### DDPM Sampling:

```python
# Initialize with pure noise
x_T = random_normal(size=(3, 64, 64))

# Iterative denoising
for t in range(T, 0, -1):
    # Predict noise at current timestep
    predicted_noise = model(x_t, t)
    
    # Denoise
    x_(t-1) = denoise_step(x_t, predicted_noise, t)
    
# Output
generated_image = x_0
```

### DDIM Sampling:
- Skips timesteps (e.g., 100 steps instead of 1000)
- Deterministic process
- Faster generation with comparable quality
---

## Comparison with GANs

| Aspect | GAN | Diffusion |
|--------|-----|-----------|
| **Steps** | 1 (noise → image) | 1000 (gradual denoising) |
| **Training** | Adversarial (unstable) | Regression (stable) |
| **Loss** | Complex (Wasserstein, GP) | Simple MSE |
| **Mode collapse** | Common | Rare |
| **Diversity** | Can be limited | Excellent |
| **Speed** | Fast (1 forward pass) | Slow (1000 passes) |
| **Quality** | Good | State-of-the-art |

---

## Strengths and Limitations

### Strengths:
1. Stable training - straightforward noise prediction objective
2. Better diversity - explores full data distribution
3. No mode collapse - cannot converge to single output
4. Controllable - amenable to guidance techniques
5. Flexible - applicable to images, audio, video, etc.

### Limitations:
1. Slower generation - requires many forward passes
2. Higher memory usage - stores all timesteps
3. Longer training - more iterations needed for convergence

---

## Implementation Overview

**Core components:**

1. **Noise scheduler**: Defines β_t values
2. **Forward diffusion**: Efficient noise addition
3. **U-Net**: Denoising network architecture
4. **Training loop**: Noise prediction at random timesteps
5. **Sampling**: Iterative denoising for generation

**Expected characteristics:**
- High diversity in generated outputs
- Stable training dynamics
- Superior quality after sufficient training
- Slower generation than single-step models
