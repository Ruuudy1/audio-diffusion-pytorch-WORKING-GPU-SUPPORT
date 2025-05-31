import torch
import soundfile as sf
from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler

def main():
    # 1) Confirm CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        return
    device = torch.device("cuda:0")
    print(f">>> Running on device: {device}")

    # 2) Build the U-Net diffusion model exactly as in the README’s “Unconditional Generator” example.
    #    Here we use the full 9-layer channel list (just as the README shows), even though it’s slower.
    #    You can reduce channels/layers if you want a quicker test.
    model = DiffusionModel(
        net_t=UNetV0,                    # U-Net V0 architecture
        in_channels=2,                   # stereo (2 channels)
        channels=[8, 32, 64, 128, 256, 512, 512, 1024, 1024],
        factors=[1, 4, 4, 4, 2, 2, 2, 2, 2],
        items=[1, 2, 2, 2, 2, 2, 2, 4, 4],
        attentions=[0, 0, 0, 0, 0, 1, 1, 1, 1],
        attention_heads=8,               # 8 attention heads
        attention_features=64,           # 64 attention features
        diffusion_t=VDiffusion,          # Variance-preserving diffusion
        sampler_t=VSampler,              # V-sampler method
        use_text_conditioning=False,     # unconditional (no text)
        use_embedding_cfg=False,         # no classifier-free guidance for text
    ).to(device)

    # 3) Create one batch of Gaussian noise on CUDA.
    #    The README’s example uses length = 2**18 = 262,144 samples,
    #    which corresponds to ~5.46 seconds at 48 kHz.
    batch_size = 1
    channels = 2
    length = 2 ** 18  # 262,144 samples
    noise = torch.randn(batch_size, channels, length, device=device)
    print(f">>> Noise tensor: shape={noise.shape}, device={noise.device}")

    # 4) Run a quick sampling pass (num_steps=20 is a reasonable trade-off for a test).
    num_steps = 20
    print(f">>> Running model.sample(noise, num_steps={num_steps}) …")
    with torch.no_grad():
        output = model.sample(noise, num_steps=num_steps)
    # output shape: [1, 2, 262_144], device=cuda:0
    print(f">>> Output tensor: shape={output.shape}, device={output.device}")
    print("✓ Sampling complete on GPU.")

    # 5) Convert output to a NumPy array on CPU and write as a 48 kHz stereo WAV.
    wav_np = output.squeeze(0).cpu().numpy().T  # shape: [262144, 2]
    sf.write("unconditional_output.wav", wav_np, samplerate=48000)
    print("✓ Saved → “unconditional_output.wav” (48 kHz stereo)")

if __name__ == "__main__":
    main()

