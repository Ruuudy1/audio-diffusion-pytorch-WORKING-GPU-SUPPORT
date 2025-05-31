import torch
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_checkpoint.py <path_to_checkpoint>")
        return
    
    checkpoint_path = sys.argv[1]
    print(f"Inspecting checkpoint: {checkpoint_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Check if it's a state dict or a full model
    if isinstance(checkpoint, dict):
        print("Checkpoint contains a state dictionary")
        # Print some keys to understand the structure
        print(f"Keys: {list(checkpoint.keys())[:10]}...")
        print(f"Total keys: {len(checkpoint.keys())}")
    else:
        print(f"Checkpoint contains an object of type: {type(checkpoint)}")
    
    print("\nDone.")

if __name__ == "__main__":
    main()