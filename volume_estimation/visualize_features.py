import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from pathlib import Path

# --- Global dictionary to store activations and hook handles ---
activations = {}
hook_handles = []

def get_activation_hook(name):
    """Creates a hook function to capture module outputs."""
    def hook(module, input, output):
        global activations
        # Detach and move to CPU to prevent VRAM buildup and for easier processing
        if isinstance(output, torch.Tensor):
            activations[name] = output.detach().cpu()
        elif isinstance(output, (list, tuple)):
            # For layers that might output multiple tensors (e.g., detection heads)
            activations[name] = [o.detach().cpu() for o in output if isinstance(o, torch.Tensor)]
        else:
            print(f"Warning: Output of module {name} (type: {module.__class__.__name__}) is "
                  f"of type {type(output)}, not a Tensor or list/tuple of Tensors. Skipping capture for this module.")
            activations[name] = None
    return hook

def visualize_and_save_feature_maps(tensor_data, base_filename, output_dir_path, max_channels_to_plot=8):
    """Visualizes feature maps from a tensor and saves them as SVG."""
    if tensor_data is None or not isinstance(tensor_data, torch.Tensor):
        print(f"Skipping visualization for {base_filename} as input is not a valid tensor.")
        return

    current_tensor = tensor_data
    if current_tensor.ndim == 3: # Might be (C, H, W) if batch_size was 1 and squeezed
        current_tensor = current_tensor.unsqueeze(0) # Add batch dimension: (1, C, H, W)

    if current_tensor.ndim != 4: # Expect (B, C, H, W)
        print(f"Skipping {base_filename}: Tensor has unexpected dimensions {current_tensor.shape}. Expected 4D (B,C,H,W).")
        return

    batch_idx = 0 # Visualize for the first image in the batch
    num_actual_channels = current_tensor.shape[1]
    channels_to_display = min(num_actual_channels, max_channels_to_plot)

    h, w = current_tensor.shape[2], current_tensor.shape[3]
    if h == 1 and w == 1:
        print(f"Feature map {base_filename} has spatial dimensions 1x1. Values (first {channels_to_display} channels):")
        for i in range(channels_to_display):
            value = current_tensor[batch_idx, i, 0, 0].item()
            print(f"  Channel {i}: {value:.4f}")
        # Skipping SVG image for 1x1 feature maps as imshow is not ideal.
        return

    if channels_to_display == 0:
        print(f"No channels to plot for {base_filename}")
        return

    cols = 4 # Number of columns in the plot grid
    rows = int(np.ceil(channels_to_display / cols))
    rows = max(1, rows) # Ensure at least one row

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3)) # Adjusted figsize
    axes = np.array(axes).flatten() # Ensure axes is always a flat array

    for i in range(channels_to_display):
        if i < len(axes):
            ax = axes[i]
            feature_map_channel = current_tensor[batch_idx, i, :, :].numpy()

            # Normalize channel for better visualization if values have large variance or not in [0,1]
            if feature_map_channel.size > 0:
                min_val, max_val = feature_map_channel.min(), feature_map_channel.max()
                if max_val > min_val: # Avoid division by zero for constant maps
                    feature_map_channel = (feature_map_channel - min_val) / (max_val - min_val)
                else: # Handle constant value map (e.g., all zeros)
                    feature_map_channel = np.ones_like(feature_map_channel) * 0.5 # Display as mid-gray

            im = ax.imshow(feature_map_channel, cmap='viridis', aspect='auto')
            ax.set_title(f'Channel {i+1}/{num_actual_channels}')
            ax.axis('off')
            # fig.colorbar(im, ax=ax, shrink=0.6) # Optional: add colorbar, can make SVGs very large

    # Hide unused subplots
    for j in range(channels_to_display, len(axes)):
        axes[j].axis('off')

    # Sanitize base_filename for use in suptitle (matplotlib doesn't like some chars)
    title_name = base_filename.replace("_", " ").title()
    fig.suptitle(title_name, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle

    output_filepath = output_dir_path / f"{base_filename}.svg"
    try:
        plt.savefig(output_filepath, format='svg', bbox_inches='tight')
        print(f"Saved: {output_filepath}")
    except Exception as e:
        print(f"Error saving {output_filepath}: {e}")
    plt.close(fig) # Close the figure to free memory

def main(args_parsed):
    global activations, hook_handles  # Declare usage of global variables

    output_dir = Path(args_parsed.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Load Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        loaded_object = torch.load(args_parsed.model_path, map_location=device)
        if isinstance(loaded_object, torch.nn.Module):
            model_to_visualize = loaded_object
        elif isinstance(loaded_object, dict) and 'model' in loaded_object and isinstance(loaded_object['model'], torch.nn.Module):
            model_to_visualize = loaded_object['model']
        # Add more robust loading if it's a state_dict (requires model class definition)
        # elif isinstance(loaded_object, dict) and ('state_dict' in loaded_object or 'model_state_dict' in loaded_object):
        #     print("Loaded a state_dict. You need to provide the model class definition to load this.")
        #     print("Please modify the script to instantiate your model class and load the state_dict.")
        #     # Example:
        #     # from your_model_module import YourModelClass
        #     # model_definition = YourModelClass(*args_for_model_init)
        #     # model_definition.load_state_dict(loaded_object['state_dict_key'])
        #     # model_to_visualize = model_definition
        #     return
        else:
            print(f"Error: Loaded object from {args_parsed.model_path} is not a recognized model or checkpoint structure.")
            return

        model_to_visualize.to(device)
        
        # <<< --- ADD THIS SNIPPET --- >>>
        # If on MPS, and model weights might be FP16, convert model to FP32
        # to match the default FP32 input tensor from T.ToTensor()
        # This specifically addresses the "MPSFloatType vs MPSHalfType" error.
        current_model_dtype = next(model_to_visualize.parameters()).dtype
        print(f"Model parameters initial dtype: {current_model_dtype}")
        if str(device) == "mps" and current_model_dtype == torch.float16:
            print("MPS device detected and model is in FP16. Converting model to FP32 for compatibility.")
            model_to_visualize = model_to_visualize.float()
            print(f"Model parameters converted to dtype: {next(model_to_visualize.parameters()).dtype}")
        # <<< --- END OF SNIPPET --- >>>
        
        model_to_visualize.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- 2. Image Input and Preprocessing ---
    try:
        img = Image.open(args_parsed.image_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: Image not found at {args_parsed.image_path}")
        return

    img_size = args_parsed.img_size
    preprocess = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(), # Scales to [0, 1]
        # Add T.Normalize here if your model requires it
    ])
    input_tensor = preprocess(img).unsqueeze(0).to(device)

    # --- 3. Register Hooks ---
    activations = {} # Clear previous activations
    for handle in hook_handles: # Clear previous handles
        handle.remove()
    hook_handles = []

    sequential_pipeline_to_hook = None
    print(f"\nAttempting to identify the sequential pipeline for hooks...")
    print(f"Type of core loaded model (model_to_visualize): {type(model_to_visualize)}")

    # Strategy 1: The loaded model itself is nn.Sequential
    if isinstance(model_to_visualize, torch.nn.Sequential):
        print("  SUCCESS: The loaded model itself is nn.Sequential.")
        sequential_pipeline_to_hook = model_to_visualize
    
    # Strategy 2: The loaded model has a 'model' attribute which is nn.Sequential
    # This is common for Ultralytics models where model_to_visualize is DetectionModel/PoseModel
    elif hasattr(model_to_visualize, 'model') and isinstance(getattr(model_to_visualize, 'model'), torch.nn.Sequential):
        print("  SUCCESS: Found nn.Sequential at 'loaded_model.model'.")
        sequential_pipeline_to_hook = getattr(model_to_visualize, 'model')
        
    # Strategy 3: The loaded model has 'model.model' which is nn.Sequential (less common for direct .pt saves)
    elif hasattr(model_to_visualize, 'model') and \
         hasattr(getattr(model_to_visualize, 'model'), 'model') and \
         isinstance(getattr(getattr(model_to_visualize, 'model'), 'model'), torch.nn.Sequential):
        print("  SUCCESS: Found nn.Sequential at 'loaded_model.model.model'.")
        sequential_pipeline_to_hook = getattr(getattr(model_to_visualize, 'model'), 'model')
        
    else:
        print(f"\n  FAILED: Could not automatically identify a suitable nn.Sequential block using common patterns.")
        print("  Please inspect the structure of your loaded model:")
        print(f"    - Type of loaded_model (model_to_visualize): {type(model_to_visualize)}")
        
        # Print attributes of model_to_visualize
        model_attrs = [attr for attr in dir(model_to_visualize) if not attr.startswith("__")]
        print(f"    - Attributes of loaded_model: {model_attrs}")

        if hasattr(model_to_visualize, 'model'):
            model_dot_model_obj = getattr(model_to_visualize, 'model')
            print(f"    - loaded_model HAS an attribute 'model'.")
            print(f"      - Type of loaded_model.model: {type(model_dot_model_obj)}")
            if isinstance(model_dot_model_obj, torch.nn.Module):
                 sub_attributes = [attr for attr in dir(model_dot_model_obj) if not attr.startswith("__")]
                 print(f"      - Attributes of loaded_model.model: {sub_attributes}")
                 if hasattr(model_dot_model_obj, 'model'):
                     model_dot_model_dot_model_obj = getattr(model_dot_model_obj, 'model')
                     print(f"      - loaded_model.model ALSO HAS an attribute 'model'.")
                     print(f"        - Type of loaded_model.model.model: {type(model_dot_model_dot_model_obj)}")
                     if isinstance(model_dot_model_dot_model_obj, torch.nn.Sequential):
                         print(f"        - This loaded_model.model.model IS an nn.Sequential. This case should have been caught by Strategy 3.")

        print("\n  If a Sequential block exists at a different path (e.g., 'model_to_visualize.backbone.layers'),")
        print("  you will need to manually modify the script to set 'sequential_pipeline_to_hook' to point to it.")
        print("  Hooks will not be registered for now.")

    if sequential_pipeline_to_hook:
        print(f"\nRegistering hooks on {len(list(sequential_pipeline_to_hook.named_children()))} modules within: {sequential_pipeline_to_hook.__class__.__name__}")
        for i, (module_name_in_seq, module_obj) in enumerate(sequential_pipeline_to_hook.named_children()):
            # Sanitize names that might be empty or problematic for filenames
            safe_module_name = module_name_in_seq if module_name_in_seq else f"module{i}"
            safe_class_name = module_obj.__class__.__name__
            
            activation_id = f"layer_{i:02d}_{safe_module_name}_{safe_class_name}"
            handle = module_obj.register_forward_hook(get_activation_hook(activation_id))
            hook_handles.append(handle)
        print(f"Successfully registered {len(hook_handles)} hooks.")
    else:
        print("No sequential pipeline found for hook registration. Further processing will be skipped.")



    # --- 4. Perform Forward Pass ---
    if not hook_handles and not sequential_pipeline_to_hook : # Skip if no hooks were set
        print("Skipping forward pass as no hooks were registered.")
    else:
        print("Performing forward pass...")
        with torch.no_grad():
            try:
                _ = model_to_visualize(input_tensor) # Output of model itself is not used here, hooks capture intermediates
            except Exception as e:
                print(f"Error during model forward pass: {e}")
                # Clean up hooks even if forward pass fails
                for h in hook_handles: h.remove()
                return
        print("Forward pass completed.")

    # --- 5. Visualize and Save Activations ---
    if not activations:
        print("No activations were captured. Check hook registration, model structure, and forward pass.")
    else:
        print("\nVisualizing and saving captured feature maps...")
        for activation_id, captured_data in activations.items():
            if isinstance(captured_data, torch.Tensor):
                visualize_and_save_feature_maps(captured_data, activation_id, output_dir, args_parsed.max_channels)
            elif isinstance(captured_data, list): # Handle outputs that are lists of tensors (e.g., Pose head)
                for item_idx, tensor_in_list_item in enumerate(captured_data):
                    list_item_id = f"{activation_id}_item_{item_idx}"
                    visualize_and_save_feature_maps(tensor_in_list_item, list_item_id, output_dir, args_parsed.max_channels)
            # `None` types (from failed captures) are handled in visualize_and_save_feature_maps

    # --- 6. Clean Up Hooks ---
    for h in hook_handles:
        h.remove()
    print(f"\nRemoved {len(hook_handles)} hooks.")
    print("Visualization script finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize CNN feature maps from a PyTorch model and save as SVG.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the .pt model file.")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the input image.")
    parser.add_argument('--output_dir', type=str, default="feature_maps_svg_output",
                        help="Directory to save SVG visualizations.")
    parser.add_argument('--img_size', type=int, default=640,
                        help="Image size (square) for preprocessing (e.g., 640 for YOLO).")
    parser.add_argument('--max_channels', type=int, default=8,
                        help="Max number of channels to visualize per layer. Use a large number (e.g., 9999) for all, "
                             "but be warned this can create many files.")

    args = parser.parse_args()
    main(args)