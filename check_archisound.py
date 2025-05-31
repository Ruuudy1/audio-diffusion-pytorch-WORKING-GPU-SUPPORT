import inspect
from archisound import ArchiSound

# Print available attributes and methods
print("ArchiSound class attributes:")
for name in dir(ArchiSound):
    if not name.startswith('_'):  # Skip private attributes
        print(f"- {name}")

# Try to create an instance and print some info
print("\nCreating ArchiSound instance...")
model = ArchiSound()
print(f"Instance created: {model}")

# Get available models
if hasattr(model, 'available_models'):
    print("\nAvailable models:")
    print(model.available_models())
elif hasattr(ArchiSound, 'available_models'):
    print("\nAvailable models (class method):")
    print(ArchiSound.available_models())
else:
    print("\nNo method to list available models found")

# Try loading the model
print("\nTrying to load the DMAE model from checkpoint...")
try:
    # Try different methods
    if hasattr(model, 'from_pretrained'):
        model.from_pretrained('dmae')
        print("Successfully loaded model via from_pretrained('dmae')")
    elif hasattr(model, 'load_checkpoint'):
        model.load_checkpoint('checkpoints/dmae.ckpt')
        print("Successfully loaded model via load_checkpoint")
except Exception as e:
    print(f"Error loading model: {e}")

# Check if model can generate audio
print("\nChecking generation methods:")
for method_name in ['generate', 'sample', 'create_audio']:
    if hasattr(model, method_name):
        method = getattr(model, method_name)
        print(f"- {method_name}: {inspect.signature(method) if callable(method) else 'Not callable'}")

print("\nDone.")
