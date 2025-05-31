import torch
import torchaudio
import os
import sys
from audio_diffusion_pytorch import DiffusionModel

def main():
    # Define parameters
    steps = 50
    length = 2**18  # ~5.5 seconds at 48kHz
    output_file = "generated_audio.wav"
    checkpoint_path = "checkpoints/dmae.ckpt"
    sr = 48000

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Load the checkpoint to inspect its contents
        print(f"Loading checkpoint {checkpoint_path} to inspect contents...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Print the structure of the checkpoint
        if isinstance(checkpoint, dict):
            print("Checkpoint contains a state dictionary")
            keys = list(checkpoint.keys())[:20]  # Show first 20 keys
            print(f"First few keys: {keys}")
            
            # Look for model structure hints in keys
            model_type_hints = {}
            for key in keys:
                parts = key.split('.')
                if len(parts) > 1:
                    root = parts[0]
                    if root not in model_type_hints:
                        model_type_hints[root] = 0
                    model_type_hints[root] += 1
            
            print("\nKey prefixes and their counts:")
            for prefix, count in model_type_hints.items():
                print(f"  {prefix}: {count} keys")
                
            # Try to create a model and load the checkpoint
            print("\nAttempting to create and load a DiffusionModel...")
            # This is a simplified default model - might need adjustment based on the checkpoint analysis
            model = DiffusionModel()
            
            try:
                model.load_state_dict(checkpoint)
                print("✓ Successfully loaded state dict directly!")
            except Exception as e:
                print(f"Error loading state dict directly: {e}")
                
                if "model." in "".join(keys):
                    print("Attempting to load with 'model.' prefix removed...")
                    # Try removing 'model.' prefix
                    new_state_dict = {}
                    for k, v in checkpoint.items():
                        if k.startswith('model.'):
                            new_state_dict[k[6:]] = v  # Remove 'model.' prefix
                        else:
                            new_state_dict[k] = v
                    
                    try:
                        model.load_state_dict(new_state_dict)
                        print("✓ Successfully loaded state dict after removing 'model.' prefix!")
                    except Exception as e:
                        print(f"Error loading state dict with prefix removed: {e}")
        else:
            print(f"Checkpoint is not a dict, but a {type(checkpoint)}")
            
    except Exception as e:
        print(f"Error during checkpoint inspection: {e}")
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
