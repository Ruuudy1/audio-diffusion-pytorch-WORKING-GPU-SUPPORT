import torch
import torchaudio
from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler

def main():
    # 1) Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is NOT available! Please ensure you have a CUDA-enabled PyTorch installed.")
        return

    device = torch.device("cuda:0")
    print(f"Using device: {device}")

    # 2) Build an unconditional DiffusionModel exactly as the script works
    model = DiffusionModel(
        net_t=UNetV0,
        in_channels=2,
        channels=[8, 16, 32, 64],
        factors=[1, 4, 4, 4],
        items=[1, 2, 2, 2],
        attentions=[0, 0, 1, 1],
        attention_heads=4,
        attention_features=32,
        diffusion_t=VDiffusion,
        sampler_t=VSampler,
        use_text_conditioning=False,     # unconditional
        use_embedding_cfg=False,
    ).to(device)

    # 3) Create a single batch of random noise on the GPU
    batch_size = 1
    in_channels = 2
    length = 2 ** 16  # 65,536 samples (approx 1.3 seconds at 48kHz)
    noise = torch.randn(batch_size, in_channels, length, device=device)
    print(f"Noise tensor: shape={noise.shape}, device={noise.device}")

    # 4) Run a quick diffusion sampling
    num_steps = 50  # More steps for better quality
    print(f"Running model.sample(noise, num_steps={num_steps}) …")
    with torch.no_grad():
        output = model.sample(noise, num_steps=num_steps)

    print(f"Output tensor: shape={output.shape}, device={output.device}")
    print("✓ Sampling complete on GPU!")

    # 5) Save to WAV file
    sample_rate = 48000
    output_path = "successful_gpu_generation.wav"
    audio_np = output.squeeze(0).cpu().numpy().T  # [length, 2]
    torchaudio.save(output_path, torch.tensor(audio_np), sample_rate)
    print(f"✓ Audio saved to {output_path}")

    # 6) Sample info
    sample_slice = output[0, 0, :10].detach().cpu().numpy()
    print("First 10 samples of channel 0 (CPU):", sample_slice)

if __name__ == "__main__":
    main()
