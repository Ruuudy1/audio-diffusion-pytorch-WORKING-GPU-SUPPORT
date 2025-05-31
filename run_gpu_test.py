import torch
from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler

def main():
    # 1) Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is NOT available! Please ensure you have a CUDA-enabled PyTorch installed.")
        return

    device = torch.device("cuda:0")
    print(f"Using device: {device}")

    # 2) Build an unconditional DiffusionModel exactly as the README shows
    #    (We’re using a small U-Net V0 with just 2 channels and 2^16 length for speed.)
    #
    #    In your README, the “Unconditional Generator” example used:
    #      in_channels=2, channels=[8,32,64,128,256,512,512,1024,1024], ...
    #
    #    For a quick sanity check, we’ll downsize:
    #      - length = 2**16 (65,536) instead of 2**18,
    #      - channels = [8,16,32,64] instead of 9 layers.
    #
    #    This runs in ~1–2 seconds on a modern GPU for one sampling pass.
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
    #    The shape must match [batch_size, in_channels, length].
    #    Here: batch_size=1, in_channels=2, length=2**16.
    batch_size = 1
    in_channels = 2
    length = 2 ** 16  # 65,536 samples (approx 1.3 seconds at 48kHz)
    noise = torch.randn(batch_size, in_channels, length, device=device)
    print(f"Noise tensor: shape={noise.shape}, device={noise.device}")

    # 4) Run a quick diffusion sampling with num_steps=10 (very fast, low quality)
    num_steps = 10
    print(f"Running model.sample(noise, num_steps={num_steps}) …")
    with torch.no_grad():
        output = model.sample(noise, num_steps=num_steps)
    # output: a FloatTensor of shape [1, 2, 65,536], on cuda:0

    print(f"Output tensor: shape={output.shape}, device={output.device}")
    print("✓ Sampling complete on GPU!")

    # 5) (Optional) If you want to verify the data, you can copy a small slice to CPU:
    sample_slice = output[0, 0, :10].detach().cpu().numpy()
    print("First 10 samples of channel 0 (CPU):", sample_slice)

    # 6) Write the output to a WAV file using scipy
    from scipy.io import wavfile
    wav_np = output.squeeze(0).cpu().numpy().T  # [length, 2]
    wavfile.write("gpu_test_output.wav", 48000, wav_np.astype('float32'))
    print("✓ Saved to gpu_test_output.wav")

if __name__ == "__main__":
    main()

