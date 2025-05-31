import torch
import torchaudio
import os
from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description='Generate audio using the diffusion model')
    parser.add_argument('--steps', type=int, default=50, help='Number of diffusion steps')
    parser.add_argument('--length', type=int, default=2**18, help='Audio length in samples')
    parser.add_argument('--output', type=str, default='generated_dmae_audio.wav', help='Output WAV filename')
    parser.add_argument('--sr', type=int, default=48000, help='Sample rate')
    args = parser.parse_args()

    # 1) Confirm CUDA is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f">>> Running on device: {device}")

    try:
        # Let's try the same pattern that works in run_gpu_test.py and run_unconditional_gpu.py
        print("Creating model...")
        
        # 2) Build a regular diffusion model
        model = DiffusionModel(
            net_t=UNetV0,
            in_channels=2,
            channels=[8, 32, 64, 128, 256, 512, 512, 1024, 1024],
            factors=[1, 4, 4, 4, 2, 2, 2, 2, 2],
            items=[1, 2, 2, 2, 2, 2, 2, 4, 4],
            attentions=[0, 0, 0, 0, 0, 1, 1, 1, 1],
            attention_heads=8,
            attention_features=64,
            diffusion_t=VDiffusion,
            sampler_t=VSampler,
            use_text_conditioning=False,
            use_embedding_cfg=False,
        ).to(device)
        
        print("Model created successfully!")
        
        # 3) Create input noise
        batch_size = 1
        channels = 2
        length = args.length
        noise = torch.randn(batch_size, channels, length, device=device)
        print(f">>> Noise tensor: shape={noise.shape}, device={noise.device}")
        
        # 4) Generate audio
        print(f">>> Generating audio with {args.steps} steps...")
        with torch.no_grad():
            audio = model.sample(noise, num_steps=args.steps)
          # 5) Save as WAV
        audio_np = audio.squeeze(0).cpu().numpy().T  # [length, 2]
        audio_tensor = torch.tensor(audio_np)
        
        # Make sure the data is in the correct shape and type
        if audio_tensor.dim() < 2:
            audio_tensor = audio_tensor.unsqueeze(0)  # Add channel dimension if needed
        
        # Ensure audio is in float32 range of [-1, 1]
        audio_tensor = audio_tensor.float()
        
        # Save using torchaudio
        try:
            torchaudio.save(args.output, audio_tensor, args.sr)
            print(f">>> Audio saved to {args.output}")
        except Exception as save_error:
            print(f"Error saving with torchaudio: {save_error}")
            
            # Fallback to scipy.io.wavfile
            try:
                from scipy.io import wavfile
                print("Falling back to scipy for saving...")
                # Scale to int16 range for scipy
                audio_int = (audio_np * 32767).astype('int16')
                wavfile.write(args.output, args.sr, audio_int)
                print(f">>> Audio saved to {args.output} using scipy")
            except Exception as scipy_error:
                print(f"Failed to save with scipy as well: {scipy_error}")
        
    except Exception as e:
        print(f"Error: {e}")
        print(f"Exception details: {str(e)}")

if __name__ == "__main__":
    main()
