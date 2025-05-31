import torch
import torchaudio
from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler

print("Imports successful")

try:
    # Check devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    print("Creating model...")
    model = DiffusionModel(
        net_t=UNetV0,
        in_channels=2,
        channels=[8, 16, 32, 64],  # Smaller model for quick testing
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
    
    print("Model created successfully")
    
    # Generate a very short sample for testing
    length = 2 ** 14  # Very short for quick test
    noise = torch.randn(1, 2, length, device=device)
    
    print("Sampling...")
    with torch.no_grad():
        audio = model.sample(noise, num_steps=10)  # Just 10 steps for speed
    
    # Save the result
    audio_np = audio.squeeze(0).cpu().numpy().T
    torchaudio.save("test_output.wav", torch.tensor(audio_np), 48000)
    print("Audio saved to test_output.wav")
    
except Exception as e:
    print(f"Error: {e}")
    print(f"Exception details: {str(e)}")
