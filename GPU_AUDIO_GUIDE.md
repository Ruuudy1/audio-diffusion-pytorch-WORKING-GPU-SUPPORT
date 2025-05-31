# Audio Diffusion on GPU: User Guide

This guide explains how to generate audio using the audio diffusion models on your GPU.

## Prerequisites

- NVIDIA GPU with CUDA support
- PyTorch installed with CUDA support
- Required packages from `setup.py`

## Quick Start: Generate Audio Using Your GPU

Run the following command to quickly generate audio:

```bash
python generate_working.py --steps 20 --output my_audio.wav
```

This will create a stereo audio file using 20 diffusion steps. The default length is around 5.5 seconds.

## Generating Better Quality Audio

For higher quality audio, increase the number of diffusion steps:

```bash
python generate_working.py --steps 50 --output high_quality.wav
```

Higher step counts (50-100) will produce smoother, more coherent audio but will take longer to generate.

## Options for generate_working.py

- `--steps`: Number of diffusion steps (default: 50). Higher values = better quality but slower generation.
- `--length`: Audio length in samples (default: 2^18 = 262,144 samples, ~5.5s at 48kHz).
- `--output`: Output WAV filename (default: generated_dmae_audio.wav).
- `--sr`: Sample rate (default: 48000 Hz).

## Using generate_gpu.py (Advanced)

This script provides more control over model complexity and duration:

```bash
python generate_gpu.py --model medium --steps 30 --duration 5 --output medium_5sec.wav
```

### Options for generate_gpu.py

- `--model`: Choose from "small", "medium", or "large" (default: medium)
  - small: Fast, less memory, lowest quality
  - medium: Balanced quality/speed
  - large: Highest quality, slower, more memory
- `--steps`: Number of diffusion steps (default: 20)
- `--duration`: Duration in seconds (default: 5.0)
- `--output`: Output filename (default: generated_audio.wav)
- `--sample-rate`: Audio sample rate (default: 48000 Hz)
- `--seed`: Random seed for reproducibility (default: none)

## Examples

### Generate a quick test audio (fast):
```bash
python generate_working.py --steps 10 --output quick_test.wav
```

### Generate high-quality audio (slower):
```bash
python generate_working.py --steps 100 --output high_quality.wav
```

### Generate a short audio clip with small model:
```bash
python generate_gpu.py --model small --steps 20 --duration 2 --output small_model_2sec.wav
```

### Generate high-quality audio with large model:
```bash
python generate_gpu.py --model large --steps 50 --duration 10 --output large_model_10sec.wav
```

## Troubleshooting

If you encounter errors:

1. Make sure PyTorch is installed with CUDA support
2. Check that your NVIDIA drivers are up-to-date
3. Try with a smaller model or fewer steps if you run out of memory
4. If file saving fails, the script will attempt to use scipy as a fallback

## Working with the Generated Audio

The output is a standard WAV file that can be played in any audio player or imported into audio editing software.
