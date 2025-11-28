import torch
import argparse
import os
import sys

def extract_model(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        sys.exit(1)

    print(f"Loading checkpoint from {input_path}...")
    try:
        checkpoint = torch.load(input_path, map_location='cpu')
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)

    model_state_dict = None

    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            print("Found 'model_state_dict' in checkpoint.")
            model_state_dict = checkpoint['model_state_dict']
        else:
            # Check if it looks like a state dict itself (keys are strings, values are tensors)
            # This is a heuristic, but common for simple saves
            is_state_dict = all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in checkpoint.items())
            if is_state_dict:
                 print("Checkpoint appears to be a model state dictionary already.")
                 model_state_dict = checkpoint
            else:
                print("Warning: Checkpoint is a dictionary but does not contain 'model_state_dict' and does not look like a state dict.")
                # We might want to save it anyway if the user knows what they are doing, 
                # but for this specific task, we are looking for the model.
                # Let's assume it might be a custom format and just save the whole thing if we can't find the key,
                # OR fail. Given the requirement is to EXTRACT, failing or warning is better.
                # Let's try to save it as is, effectively just copying/renaming, but warning the user.
                print("Saving content as is.")
                model_state_dict = checkpoint
    else:
        print("Checkpoint is not a dictionary. Assuming it is the model object or state dict directly.")
        model_state_dict = checkpoint

    print(f"Saving model state dictionary to {output_path}...")
    torch.save(model_state_dict, output_path)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract model state dictionary from a training checkpoint.")
    parser.add_argument("input_checkpoint", help="Path to the input checkpoint file.")
    parser.add_argument("--output", "-o", default="extracted_model.pth", help="Path to save the extracted model (default: extracted_model.pth).")

    args = parser.parse_args()

    extract_model(args.input_checkpoint, args.output)
