import torch
import torchaudio
import os
import argparse
import sys
from archisound import ArchiSound

def main():
    parser = argparse.ArgumentParser(description='Generate audio using the ArchiSound model')
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
        print("Loading ArchiSound model...")
        # Create the ArchiSound model and move it to the correct device
        model = ArchiSound(device=device)
        
        # Set up the model with the checkpoint
        print(f"Loading weights from {args.checkpoint}...")
        model.load_checkpoint(args.checkpoint)
        print(f"✓ Model loaded successfully")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Exception details: {str(e)}")
        return
    
    print(f"Generating {args.length/args.sr:.2f} seconds of audio ({args.length} samples)")
    print(f"Using {args.steps} diffusion steps")
    
    # Generate audio
    print("Generating audio...")
    try:
        # Generate audio according to ArchiSound's API
        audio = model.generate(
            seconds=args.length/args.sr,
            num_steps=args.steps
        )
        
        # Save the generated audio
        torchaudio.save(args.output, audio, args.sr)
        print(f"✓ Saved audio to {args.output}")
    except Exception as e:
        print(f"Error generating audio: {e}")
        print(f"Exception details: {str(e)}")

if __name__ == "__main__":
    main()
