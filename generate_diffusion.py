import torch
import torchaudio
from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler
import argparse

def main():
    parser = argparse.ArgumentParser(description='Generate audio using the diffusion model')
    parser.add_argument('--steps', type=int, default=50, help='Number of diffusion steps')
    parser.add_argument('--length', type=int, default=2**18, help='Audio length in samples')
    parser.add_argument('--output', type=str, default='generated_audio.wav', help='Output WAV filename')
    parser.add_argument('--sr', type=int, default=48000, help='Sample rate')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is NOT available! Please ensure you have a CUDA-enabled PyTorch installed.")
        return

    device = torch.device("cuda:0")
    print(f"Using device: {device}")

    # Create a model - using the same architecture as in run_gpu_test.py but with more channels
    print("Creating diffusion model...")
    model = DiffusionModel(
        net_t=UNetV0,
        in_channels=2,
        channels=[8, 32, 64, 128, 256, 512, 512, 1024, 1024], # Full-scale model
        factors=[1, 4, 4, 4, 2, 2, 2, 2, 2],
        items=[1, 2, 2, 2, 2, 2, 2, 4, 4],
        attentions=[0, 0, 1, 1, 1, 1, 1, 1, 1],
        attention_heads=8,
        attention_features=64,
        diffusion_t=VDiffusion,
        sampler_t=VSampler,
        use_text_conditioning=False,     # unconditional
        use_embedding_cfg=False,
    ).to(device)

    # Create input noise
    batch_size = 1
    in_channels = 2
    length = args.length  # Default: 2**18 (262,144 samples ≈ 5.5 seconds at 48kHz)
    noise = torch.randn(batch_size, in_channels, length, device=device)
    print(f"Noise tensor: shape={noise.shape}, device={noise.device}")

    # Generate audio
    print(f"Generating audio with {args.steps} diffusion steps...")
    with torch.no_grad():
        output = model.sample(noise, num_steps=args.steps)

    print(f"Output tensor: shape={output.shape}, device={output.device}")
    print("✓ Sampling complete on GPU!")

    # Convert to audio format and save as WAV
    wav = output.squeeze(0).cpu()  # Remove batch dimension and move to CPU
    torchaudio.save(args.output, wav, args.sr)
    print(f"✓ Saved audio to {args.output}")
    
    # Optional: Print a small sample of the generated audio
    sample_slice = output[0, 0, :10].detach().cpu().numpy()
    print("First 10 samples of channel 0 (CPU):", sample_slice)

if __name__ == "__main__":
    main()
