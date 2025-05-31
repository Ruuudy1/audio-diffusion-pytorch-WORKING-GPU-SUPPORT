import torch
import torchaudio
import os
import argparse
import sys
from archisound import ArchiSound

def main():
    parser = argparse.ArgumentParser(description='Generate audio using the diffusion model')
    parser.add_argument('--steps', type=int, default=50, help='Number of diffusion steps')
    parser.add_argument('--length', type=int, default=2**18, help='Audio length in samples')
    parser.add_argument('--output', type=str, default='generated_audio.wav', help='Output WAV filename')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/dmae.ckpt', help='Model checkpoint path')
    parser.add_argument('--sr', type=int, default=48000, help='Sample rate')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
      # Load model
    try:
        print("Creating DMAE model...")
        # Create the DMAE model
        model = DMAE(
            in_channels=2,
            channels=64,
            patch_blocks=1,
            patch_factor=8,
            latent_dim=512,
            multipliers=[1, 2, 4, 8, 8],
            factors=[1, 4, 4, 4, 2],
            num_blocks=[2, 2, 2, 2, 2],
            attn_blocks=[0, 0, 0, 1, 1],
            use_vqembed=False
        ).to(device)
        
        # Now load the state dictionary from checkpoint
        print(f"Loading weights from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        # Load state dict
        # This model expects the state dict keys to start with 'model.'
        if isinstance(checkpoint, dict) and any(key.startswith('model.') for key in checkpoint.keys()):
            model.load_state_dict(checkpoint)
            print(f"✓ Model loaded successfully")
        else:
            print("Checkpoint format not recognized. Expected keys starting with 'model.'")
            return
            
        # Make sure the model is in evaluation mode
        model.eval()
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    print(f"Generating {args.length/args.sr:.2f} seconds of audio ({args.length} samples)")
    print(f"Using {args.steps} diffusion steps")
    
    # Create input noise
    noise = torch.randn(1, 2, args.length, device=device)
    
    # Generate audio
    print("Generating audio...")
    with torch.no_grad():
        audio = model.sample(noise, num_steps=args.steps)
    
    # Convert to numpy and save as WAV
    audio_np = audio.squeeze(0).cpu().numpy().T  # [length, channels]
    torchaudio.save(args.output, torch.tensor(audio_np), args.sr)
    print(f"✓ Saved audio to {args.output}")

if __name__ == "__main__":
    main()