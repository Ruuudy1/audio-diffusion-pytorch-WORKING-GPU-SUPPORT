import torch
import argparse
from scipy.io import wavfile
from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate audio using diffusion model on GPU")
    parser.add_argument("--steps", type=int, default=50, help="Number of diffusion steps (more = higher quality, slower)")
    parser.add_argument("--length", type=int, default=2**17, help="Audio length in samples (powers of 2 work best)")
    parser.add_argument("--output", type=str, default="generated_audio.wav", help="Output filename")
    parser.add_argument("--sr", type=int, default=48000, help="Sample rate")
    parser.add_argument("--quality", choices=["low", "medium", "high"], default="medium", 
                        help="Model quality/size - low is fastest")
    args = parser.parse_args()

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA is NOT available! Please ensure you have a CUDA-enabled PyTorch installed.")
        return

    device = torch.device("cuda:0")
    print(f"Using device: {device}")
    
    # Configure model size based on quality setting
    if args.quality == "low":
        channels = [8, 16, 32, 64]  # Small, fast model
        factors = [1, 4, 4, 4]
        items = [1, 2, 2, 2]
        attentions = [0, 0, 1, 1]
        attn_heads = 4
        attn_features = 32
    elif args.quality == "medium":
        channels = [8, 32, 64, 128, 256, 512]  # Medium model
        factors = [1, 4, 4, 4, 2, 2]
        items = [1, 2, 2, 2, 2, 2]
        attentions = [0, 0, 0, 1, 1, 1]
        attn_heads = 8
        attn_features = 64
    else:  # high
        channels = [8, 32, 64, 128, 256, 512, 512, 1024, 1024]  # Full model from README
        factors = [1, 4, 4, 4, 2, 2, 2, 2, 2]
        items = [1, 2, 2, 2, 2, 2, 2, 4, 4]
        attentions = [0, 0, 0, 0, 0, 1, 1, 1, 1]
        attn_heads = 8
        attn_features = 64

    # Create the model
    print(f"Creating {args.quality} quality diffusion model...")
    model = DiffusionModel(
        net_t=UNetV0,
        in_channels=2,  # stereo audio
        channels=channels,
        factors=factors,
        items=items,
        attentions=attentions,
        attention_heads=attn_heads,
        attention_features=attn_features,
        diffusion_t=VDiffusion,
        sampler_t=VSampler,
        use_text_conditioning=False,
        use_embedding_cfg=False,
    ).to(device)
    print("Model created successfully")

    # Create the input noise
    print(f"Creating noise with length {args.length} samples...")
    noise = torch.randn(1, 2, args.length, device=device)

    # Run the sampling process
    print(f"Running sampling with {args.steps} steps (this may take a while)...")
    with torch.no_grad():
        audio = model.sample(noise, num_steps=args.steps)

    print(f"Sampling complete! Output tensor shape: {audio.shape}")
    
    # Convert and save as WAV
    print(f"Saving to {args.output}...")
    wav_np = audio.squeeze(0).cpu().numpy().T  # [length, 2]
    wavfile.write(args.output, args.sr, wav_np.astype('float32'))
    
    # Show info
    duration = args.length / args.sr
    print(f"✓ Audio generation complete!")
    print(f"  - File: {args.output}")
    print(f"  - Duration: {duration:.2f} seconds ({args.length} samples at {args.sr} Hz)")
    print(f"  - Quality: {args.quality} ({args.steps} diffusion steps)")

if __name__ == "__main__":
    main()
