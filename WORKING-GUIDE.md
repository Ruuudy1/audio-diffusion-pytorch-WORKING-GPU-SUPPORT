# Using the Audio Diffusion PyTorch Models

This guide provides step-by-step instructions for generating audio using the audio-diffusion-pytorch models on your GPU.

## Prerequisites
CUDA-capable NVIDIA GPU with drivers installed
Python 3.7 or newer
Repository cloned to your local machine

# Step 1: Activate the Environment
```python
# Navigate to the repository directory
cd ~/spring2025/cse253R/audio-diffusion-pytorch # my specific path

# Activate the Python environment
source mousai_env/bin/activate  # For Windows: mousai_env\Scripts\activate
```

# Step 2: Verify GPU Functionality
```python
# Run the test script to ensure GPU is working
python run_gpu_test.py
```

You should see output confirming your model is running on the GPU and a short audio sample is generated.

# Step 3: Generate Basic Audio
```python
# Generate audio with default settings
python generate_working.py --output my_audio.wav
```

This generates approximately 5.5 seconds of stereo audio using 50 diffusion steps.

# Step 4: Adjust Quality Settings
```python
# Low quality (faster)
python generate_working.py --steps 20 --output quick_audio.wav

# Medium quality (default)
python generate_working.py --steps 50 --output medium_audio.wav

# High quality (slower)
python generate_working.py --steps 100 --output high_quality_audio.wav

# Ultra quality (much slower)
python generate_working.py --steps 200 --output ultra_audio.wav
```

The quality is controlled by the number of diffusion steps:

# Step 5: Adjust Audio Duration
```python
# Short clip (1.4 seconds at 48kHz)
python generate_working.py --length 65536 --output short.wav

# Standard clip (5.5 seconds at 48kHz)
python generate_working.py --length 262144 --output standard.wav

# Long clip (10.9 seconds at 48kHz)
python generate_working.py --length 524288 --output long.wav
```

The audio length is controlled by the --length parameter:

Note: Length must be a power of 2 (typically 2^16 to 2^19).

# Step 6: Combining Parameters
```python
# Generate a long, high-quality clip
python generate_working.py --steps 150 --length 524288 --output long_high_quality.wav
```

You can combine duration and quality parameters:

# Step 7: Listen to Your Generated Audio
Open the generated WAV files with any audio player.

## Common Issues and Solutions

### Out of Memory Error:

Try reducing the --length parameter
Use fewer diffusion steps with --steps
Close other GPU-intensive applications
Slow Generation:

### Lower the number of steps
Generate shorter clips
Use a smaller model

### Audio Quality Issues:

Increase the number of steps
Try multiple generations (results vary due to random initialization)

## Advanced Usage
For more advanced configurations, edit the generate_working.py file and modify the model parameters:

```python
model = DiffusionModel(
    net_t=UNetV0,
    in_channels=2,  # stereo audio
    channels=[8, 32, 64, 128, 256, 512, 512, 1024, 1024],  # adjust for quality/speed
    factors=[1, 4, 4, 4, 2, 2, 2, 2, 2],  # downsampling factors
    items=[1, 2, 2, 2, 2, 2, 2, 4, 4],  # block multipliers
    attentions=[0, 0, 0, 1, 1, 1, 1, 1, 1],  # attention layers
    attention_heads=8,  # attention heads
    attention_features=64,  # attention features
    diffusion_t=VDiffusion,  # diffusion algorithm
    sampler_t=VSampler,  # sampler algorithm
).to(device)
```