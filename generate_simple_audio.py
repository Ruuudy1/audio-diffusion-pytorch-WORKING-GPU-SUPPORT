import torch
import torchaudio

# 1. Define paths
checkpoint_path = "checkpoints/dmae.ckpt"
output_file = "dmae_audio.wav"
sample_rate = 48000
num_steps = 20

# 2. Set up device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

try:
    # 3. Create vanilla model
    from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler
    
    print("Creating basic diffusion model...")
    model = DiffusionModel(
        net_t=UNetV0,
        in_channels=2,
        channels=[8, 16, 32, 64],  # Small for testing
        factors=[1, 4, 4, 4],
        items=[1, 2, 2, 2],
        attentions=[0, 0, 1, 1],
        attention_heads=4,
        attention_features=32,
        diffusion_t=VDiffusion,
        sampler_t=VSampler,
    ).to(device)
    
    # 4. Generate with vanilla model
    length = 2 ** 16  # Short for testing ~1.4s at 48kHz
    print(f"Generating audio ({length} samples)...")
    noise = torch.randn(1, 2, length, device=device)
    
    with torch.no_grad():
        audio = model.sample(noise, num_steps=num_steps)
    
    # 5. Save audio
    audio_np = audio.squeeze(0).cpu().numpy().T
    torchaudio.save(output_file, torch.tensor(audio_np), sample_rate)
    print(f"Audio saved to {output_file}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
