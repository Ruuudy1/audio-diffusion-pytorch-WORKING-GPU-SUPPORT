import torch
import torchaudio
import os
import argparse
import sys
from archisound import ArchiSound

def main():
    parser = argparse.ArgumentParser(description='Generate audio using ArchiSound')
    parser.add_argument('--steps', type=int, default=50, help='Number of diffusion steps')
    parser.add_argument('--length', type=float, default=5.0, help='Audio length in seconds')
    parser.add_argument('--output', type=str, default='generated_audio.wav', help='Output WAV filename')
    parser.add_argument('--model', type=str, default='dmae-atc', help='Model name (default: dmae-atc)')
    parser.add_argument('--sr', type=int, default=48000, help='Sample rate (default: 48000)')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Create ArchiSound model
        print(f"Creating ArchiSound model and loading pretrained '{args.model}'...")
        model = ArchiSound(device=device)
        
        # Load pretrained model
        model.from_pretrained(args.model)
        print(f"✓ Model '{args.model}' loaded successfully")
        
        # Generate audio
        print(f"Generating {args.length} seconds of audio with {args.steps} diffusion steps...")
        audio = model.generate(seconds=args.length, num_steps=args.steps)
        
        # Save the audio
        print(f"Saving audio to {args.output}...")
        if hasattr(model, 'save_audio'):
            # Use model's save_audio method if available
            model.save_audio(audio, args.output)
        else:
            # Fallback to torchaudio.save
            sr = getattr(model, 'sample_rate', args.sr)
            torchaudio.save(args.output, audio, sr)
        
        print(f"✓ Audio saved to {args.output}")
        
    except Exception as e:
        print(f"Error: {e}")
        print(f"Stack trace:", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
