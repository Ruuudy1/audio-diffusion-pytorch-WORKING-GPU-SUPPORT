import torch
import argparse
import numpy as np
from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler

MODEL_PRESETS = {
    "small": {
        "channels": [8, 16, 32, 64],
        "factors": [1, 4, 4, 4],
        "items": [1, 2, 2, 2],
        "attentions": [0, 0, 1, 1],
        "attention_heads": 4,
        "attention_features": 32
    },
    "medium": {
        "channels": [8, 32, 64, 128, 256, 512],
        "factors": [1, 4, 4, 4, 2, 2],
        "items": [1, 2, 2, 2, 2, 2],
        "attentions": [0, 0, 0, 1, 1, 1],
        "attention_heads": 8,
        "attention_features": 64
    },
    "large": {
        "channels": [8, 32, 64, 128, 256, 512, 512, 1024, 1024],
        "factors": [1, 4, 4, 4, 2, 2, 2, 2, 2],
        "items": [1, 2, 2, 2, 2, 2, 2, 4, 4],
        "attentions": [0, 0, 0, 0, 0, 1, 1, 1, 1],
        "attention_heads": 8,
        "attention_features": 64
    }
}

def get_duration_text(seconds):
    """Format seconds as minutes and seconds."""
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes > 0:
        return f"{minutes}m {secs:.1f}s"
    else:
        return f"{secs:.1f}s"

def main():
    parser = argparse.ArgumentParser(description='Generate audio with GPU acceleration')
    parser.add_argument('--output', type=str, default='generated_audio.wav', 
                       help='Output WAV filename')
    parser.add_argument('--steps', type=int, default=20,
                       help='Number of diffusion steps (more = higher quality, slower generation)')
    parser.add_argument('--duration', type=float, default=5.0,
                       help='Duration in seconds')
    parser.add_argument('--model', type=str, choices=["small", "medium", "large"], default="medium",
                       help='Model size preset')
    parser.add_argument('--sample-rate', type=int, default=48000,
                       help='Audio sample rate')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"Random seed set to {args.seed}")

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Calculate length in samples
    length = int(args.duration * args.sample_rate)
    length = 2 ** (length - 1).bit_length()  # Round to nearest power of 2
    actual_duration = length / args.sample_rate
    
    print(f"Generating {get_duration_text(actual_duration)} of audio ({length} samples)")
    print(f"Using {args.model} model with {args.steps} diffusion steps")
    
    # Get model preset
    preset = MODEL_PRESETS[args.model]
    
    try:
        # Build model
        print("Creating model...")
        model = DiffusionModel(
            net_t=UNetV0,
            in_channels=2,  # Stereo audio
            channels=preset["channels"],
            factors=preset["factors"],
            items=preset["items"],
            attentions=preset["attentions"],
            attention_heads=preset["attention_heads"],
            attention_features=preset["attention_features"],
            diffusion_t=VDiffusion,
            sampler_t=VSampler,
            use_text_conditioning=False,
            use_embedding_cfg=False,
        ).to(device)
        
        # Create noise
        print("Generating noise...")
        noise = torch.randn(1, 2, length, device=device)
        
        # Generate audio
        print(f"Running diffusion sampling with {args.steps} steps...")
        with torch.no_grad():
            audio = model.sample(noise, num_steps=args.steps)
            
        # Save to file
        print("Processing audio...")
        audio_np = audio.squeeze(0).cpu().numpy().T
        
        # Save using scipy (most reliable)
        from scipy.io import wavfile
        print(f"Saving to {args.output}...")
        audio_int = (audio_np * 32767).astype('int16')  # Convert to int16
        wavfile.write(args.output, args.sample_rate, audio_int)
        
        print(f"✓ Successfully generated {get_duration_text(actual_duration)} of audio!")
        print(f"✓ Saved to {args.output}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
