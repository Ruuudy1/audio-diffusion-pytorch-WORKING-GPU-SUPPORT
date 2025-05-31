import torch
from archisound import ArchiSound

# Check if CUDA is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

try:
    # Load model
    print("Creating ArchiSound model...")
    model = ArchiSound(device=device)
    
    # Try to load from pretrained checkpoint
    print("Loading from pretrained model...")
    model.from_pretrained("dmae-atc")
    print("Successfully loaded pretrained model!")
    
    # Generate a 5 second clip
    print("Generating audio...")
    audio = model.generate(seconds=5.0, num_steps=50)
    
    # Save as .wav file
    model.save_audio(audio, "generated_audio.wav")
    print("✓ Audio saved to generated_audio.wav")
except Exception as e:
    print(f"Error: {e}")
    print(f"Exception details: {str(e)}")
