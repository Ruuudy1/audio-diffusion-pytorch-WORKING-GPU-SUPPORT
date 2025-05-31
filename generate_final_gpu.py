import torch
import torchaudio
import argparse
from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler

def main():
    parser = argparse.ArgumentParser(description='Generate audio using diffusion model')
    parser.add_argument('--steps', type=int, default=50, help='Number of diffusion steps')
    parser.add_argument('--length', type=int, default=2**18, help='Audio length in samples')
    parser.add_argument('--output', type=str, default='generated_audio_final.wav', help='Output file name')
    parser.add_argument('--sr', type=int, default=48000, help='Sample rate')
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("CUDA is NOT available! Please ensure you have a CUDA-enabled PyTorch installed.")
        return

    device = torch.device("cuda:0")
    print(f"Using device: {device}")

    try:
        # Create a model - using the same parameters as run_gpu_test.py which we know works
        print("Creating diffusion model...")
        model = DiffusionModel(
            net_t=UNetV0,
            in_channels=2,
            channels=[8, 16, 32, 64],  # Small for faster generation
            factors=[1, 4, 4, 4],
            items=[1, 2, 2, 2],
            attentions=[0, 0, 1, 1],
            attention_heads=4,
            attention_features=32,
            diffusion_t=VDiffusion,
            sampler_t=VSampler,
            use_text_conditioning=False,
            use_embedding_cfg=False,
        ).to(device)
        print("Model created successfully!")

        # Create noise tensor
        batch_size = 1
        in_channels = 2
        length = args.length  # Use the length from arguments
        noise = torch.randn(batch_size, in_channels, length, device=device)
        print(f"Noise tensor: shape={noise.shape}, device={noise.device}")

        # Run sampling
        print(f"Running model.sample() with {args.steps} steps...")
        with torch.no_grad():
            output = model.sample(noise, num_steps=args.steps)

        print(f"Output tensor: shape={output.shape}, device={output.device}")
        print("✓ Sampling complete on GPU!")

        # Save the output as WAV
        print(f"Saving audio to {args.output}...")
        audio_np = output.squeeze(0).cpu().numpy().T  # [length, 2]
        torchaudio.save(args.output, torch.tensor(audio_np), args.sr)
        print(f"✓ File saved successfully! Duration: {length/args.sr:.2f} seconds")
        
        # Show first few samples
        sample_slice = output[0, 0, :10].detach().cpu().numpy()
        print("First 10 samples of channel 0:", sample_slice)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
