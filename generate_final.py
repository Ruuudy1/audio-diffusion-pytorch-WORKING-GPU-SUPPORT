import torch
import torchaudio
import argparse

def main():
    parser = argparse.ArgumentParser(description='Generate audio using diffusion model')
    parser.add_argument('--steps', type=int, default=50, help='Number of diffusion steps')
    parser.add_argument('--length', type=int, default=2**18, help='Audio length in samples')
    parser.add_argument('--output', type=str, default='final_generated.wav', help='Output file name')
    parser.add_argument('--sr', type=int, default=48000, help='Sample rate')
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Import model classes
    from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler
    
    print("Creating diffusion model...")
    model = DiffusionModel(
        net_t=UNetV0,
        in_channels=2,  # Stereo audio
        # Use full-size model as in README for higher quality
        channels=[8, 32, 64, 128, 256, 512, 512, 1024, 1024],
        factors=[1, 4, 4, 4, 2, 2, 2, 2, 2],
        items=[1, 2, 2, 2, 2, 2, 2, 4, 4],
        attentions=[0, 0, 0, 0, 0, 1, 1, 1, 1],
        attention_heads=8,
        attention_features=64,
        diffusion_t=VDiffusion,
        sampler_t=VSampler,
        use_text_conditioning=False,  # No text conditioning
        use_embedding_cfg=False,
    ).to(device)
    
    # Create noise input
    print(f"Creating noise with length {args.length} samples...")
    noise = torch.randn(1, 2, args.length, device=device)
    
    # Generate audio
    print(f"Generating audio with {args.steps} diffusion steps (this may take a while)...")
    with torch.no_grad():
        audio = model.sample(noise, num_steps=args.steps)
    
    # Save to file
    audio_np = audio.squeeze(0).cpu().numpy().T  # Shape: [length, 2]
    torchaudio.save(args.output, torch.tensor(audio_np), args.sr)
    print(f"✓ Successfully saved audio to {args.output}")
    
    # Calculate duration
    duration_seconds = args.length / args.sr
    print(f"Audio duration: {duration_seconds:.2f} seconds")
    
    print("Done!")

if __name__ == "__main__":
    main()
