# """
# Advanced WorldModel Debug Visualization
# This version tries multiple approaches to get proper reconstructions
# """

# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
# import os
# import yaml
# import argparse
# from pathlib import Path
# from gymnasium.spaces import Box, Dict

# from training_wm_v2 import (
#     Config,
#     apply_size_override,
#     setup_config_for_worldmodel,  
#     load_checkpoint
# )

# from models import WorldModel
# from hdf5_converter_v2 import convert_hdf5_to_dreamer


# def analyze_tensor_stats(tensor, name):
#     """Print detailed statistics about a tensor"""
#     if torch.is_tensor(tensor):
#         tensor_np = tensor.detach().cpu().numpy()
#     else:
#         tensor_np = tensor
    
#     print(f"\n{name} Statistics:")
#     print(f"  Shape: {tensor_np.shape}")
#     print(f"  Range: [{tensor_np.min():.4f}, {tensor_np.max():.4f}]")
#     print(f"  Mean: {tensor_np.mean():.4f}, Std: {tensor_np.std():.4f}")
#     print(f"  Unique values: {len(np.unique(tensor_np))}")
    
#     # Check for common ranges
#     if tensor_np.min() >= 0 and tensor_np.max() <= 1:
#         print("  -> Appears to be in [0,1] range")
#     elif tensor_np.min() >= -1 and tensor_np.max() <= 1:
#         print("  -> Appears to be in [-1,1] range")
#     elif tensor_np.min() >= 0 and tensor_np.max() <= 255:
#         print("  -> Appears to be in [0,255] range")
#     else:
#         print("  -> Custom range")


# def try_multiple_normalizations(tensor, name):
#     """Try different normalization approaches and return the best looking one"""
#     if torch.is_tensor(tensor):
#         tensor = tensor.detach().cpu().numpy()
    
#     tensor = tensor.copy()
    
#     print(f"\nTrying different normalizations for {name}:")
    
#     # Method 1: Direct clipping to [0,1]
#     method1 = np.clip(tensor, 0, 1)
#     print(f"  Method 1 (clip to [0,1]): range [{method1.min():.4f}, {method1.max():.4f}]")
    
#     # Method 2: Min-max normalization
#     if tensor.max() != tensor.min():
#         method2 = (tensor - tensor.min()) / (tensor.max() - tensor.min())
#         print(f"  Method 2 (min-max norm): range [{method2.min():.4f}, {method2.max():.4f}]")
#     else:
#         method2 = np.zeros_like(tensor)
#         print(f"  Method 2 (min-max norm): constant tensor, using zeros")
    
#     # Method 3: Sigmoid activation (if model outputs logits)
#     method3 = 1 / (1 + np.exp(-tensor))  # sigmoid
#     print(f"  Method 3 (sigmoid): range [{method3.min():.4f}, {method3.max():.4f}]")
    
#     # Method 4: Tanh to [0,1] 
#     method4 = (np.tanh(tensor) + 1) / 2
#     print(f"  Method 4 (tanh->01): range [{method4.min():.4f}, {method4.max():.4f}]")
    
#     # Method 5: Add offset and scale (if model learned a different range)
#     method5 = tensor + 0.5  # Common if model outputs around [-0.5, 0.5]
#     method5 = np.clip(method5, 0, 1)
#     print(f"  Method 5 (+0.5 clip): range [{method5.min():.4f}, {method5.max():.4f}]")
    
#     return {
#         'clip_01': method1,
#         'minmax': method2, 
#         'sigmoid': method3,
#         'tanh': method4,
#         'offset': method5,
#         'raw': tensor
#     }


# def save_comparison_methods(original, reconstructed_methods, output_dir, seq_idx):
#     """Save comparison grids for different reconstruction methods"""
    
#     # Normalize original
#     if torch.is_tensor(original):
#         original = original.detach().cpu().numpy()
    
#     if original.max() > 1:
#         original = original / 255.0
#     original = np.clip(original, 0, 1)
    
#     # Ensure channel order is correct
#     if original.ndim == 4 and original.shape[1] == 3:
#         original = np.transpose(original, (0, 2, 3, 1))
    
#     # Create comparison for each method
#     for method_name, reconstructed in reconstructed_methods.items():
#         if reconstructed.ndim == 4 and reconstructed.shape[1] == 3:
#             reconstructed = np.transpose(reconstructed, (0, 2, 3, 1))
        
#         # Create grid comparing first 8 frames
#         num_frames = min(8, original.shape[0])
#         fig, axes = plt.subplots(2, num_frames, figsize=(num_frames*3, 6))
        
#         if num_frames == 1:
#             axes = axes.reshape(2, 1)
        
#         fig.suptitle(f'Method: {method_name} - Original vs Reconstructed', fontsize=16)
        
#         for i in range(num_frames):
#             # Original
#             axes[0, i].imshow(original[i])
#             axes[0, i].set_title(f'Original {i}')
#             axes[0, i].axis('off')
            
#             # Reconstructed
#             axes[1, i].imshow(reconstructed[i])
#             axes[1, i].set_title(f'Recon {i}')
#             axes[1, i].axis('off')
        
#         plt.tight_layout()
        
#         save_path = os.path.join(output_dir, f'method_{method_name}_seq_{seq_idx}.png')
#         fig.savefig(save_path, dpi=150, bbox_inches='tight')
#         plt.close(fig)
#         print(f"Saved {method_name} comparison: {save_path}")


# def advanced_worldmodel_debug(config_path, hdf5_path, checkpoint_path, size='1m', 
#                             output_dir='./advanced_debug', sequence_idx=0):
#     """Advanced debugging with multiple reconstruction attempts"""
    
#     print(f"=== ADVANCED WORLDMODEL DEBUG ===")
#     print(f"Loading model: {checkpoint_path}")
    
#     # Create output directory
#     Path(output_dir).mkdir(parents=True, exist_ok=True)
    
#     # Load config
#     with open(config_path, 'r') as f:
#         config = yaml.safe_load(f)

#     size_key = f"size{size}"
#     apply_size_override(config, size_key)
#     cfg_dict = config['defaults'].copy()
    
#     size_overrides = config.get(size_key, {})
#     cfg_dict = setup_config_for_worldmodel(cfg_dict, size_overrides)

#     # Load data
#     data = convert_hdf5_to_dreamer(hdf5_path)
#     analyze_tensor_stats(data['image'], "Original Data")
    
#     # Store truly original images
#     original_images = data['image'].clone()
    
#     # Normalize for model
#     if data['image'].max() > 1.0:
#         data['image'] = data['image'] / 255.0
    
#     # Add missing data
#     if 'reward' not in data:
#         data['reward'] = torch.zeros(data['action'].shape[:2], dtype=torch.float32)
#     if 'discount' not in data:
#         data['discount'] = torch.ones(data['action'].shape[:2], dtype=torch.float32)
    
#     # Setup model
#     actual_action_dim = data['action'].shape[-1]
#     cfg_dict['num_actions'] = actual_action_dim
#     cfg_dict['batch_size'] = 16
#     cfg_dict['batch_length'] = 64

#     cfg = Config(**cfg_dict)
#     device = cfg_dict.get('device', 'cuda')

#     image_shape = data['image'].shape[2:]
#     obs_space = Dict({'image': Box(low=0, high=1, shape=image_shape, dtype=np.float32)})
#     act_space = Box(low=-1.0, high=1.0, shape=(actual_action_dim,), dtype=np.float32)

#     # Initialize and load model
#     wm = WorldModel(obs_space, act_space, 0, cfg).to(device)
#     epoch, metrics = load_checkpoint(wm, checkpoint_path)
#     print(f"\nLoaded model from epoch {epoch}")
#     print("Training metrics:", {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in metrics.items()})
    
#     wm.eval()

#     # Process sequence
#     seq_idx = sequence_idx
#     batch_length = min(cfg_dict['batch_length'], data['image'].shape[1])
    
#     original_seq = original_images[seq_idx, :batch_length]
#     analyze_tensor_stats(original_seq, "Original Sequence")
    
#     # Prepare batch
#     batch_data = {
#         'image': data['image'][seq_idx:seq_idx+1, :batch_length].to(device),
#         'action': data['action'][seq_idx:seq_idx+1, :batch_length].to(device),
#         'reward': data['reward'][seq_idx:seq_idx+1, :batch_length].to(device),
#         'is_first': data['is_first'][seq_idx:seq_idx+1, :batch_length].to(device),
#         'is_terminal': data['is_terminal'][seq_idx:seq_idx+1, :batch_length].to(device),
#         'discount': data['discount'][seq_idx:seq_idx+1, :batch_length].to(device),
#     }

#     print(f"\n=== MODEL FORWARD PASS ===")
    
#     with torch.no_grad():
#         # Step-by-step model forward pass
#         print("1. Preprocessing...")
#         preprocessed = wm.preprocess(batch_data)
#         analyze_tensor_stats(preprocessed['image'], "Preprocessed Image")
        
#         print("2. Encoding...")
#         embed = wm.encoder(preprocessed)
#         analyze_tensor_stats(embed, "Embeddings")
        
#         print("3. Dynamics...")
#         post, prior = wm.dynamics.observe(embed, preprocessed["action"], preprocessed["is_first"])
#         feat = wm.dynamics.get_feat(post)
#         analyze_tensor_stats(feat, "Features")
        
#         print("4. Decoding...")
#         decoder_pred = wm.heads["decoder"](feat)
#         recon_dist = decoder_pred["image"]
        
#         # Handle different distribution types
#         if hasattr(recon_dist, 'mode'):
#             reconstructed = recon_dist.mode()
#             print("Using distribution mode")
#         elif hasattr(recon_dist, 'mean'):
#             reconstructed = recon_dist.mean()
#             print("Using distribution mean")
#         elif hasattr(recon_dist, 'sample'):
#             reconstructed = recon_dist.sample()
#             print("Using distribution sample")
#         else:
#             reconstructed = recon_dist
#             print("Using raw output")
        
#         analyze_tensor_stats(reconstructed, "Raw Reconstruction")
        
#         # Check what type of distribution this is
#         print(f"Decoder output type: {type(recon_dist)}")
#         if hasattr(recon_dist, '__dict__'):
#             print(f"Distribution attributes: {list(recon_dist.__dict__.keys())}")
        
#         # Remove batch dimension
#         reconstructed = reconstructed[0].cpu()  # [T, H, W, C]
        
#         print(f"\n=== TRYING DIFFERENT NORMALIZATIONS ===")
#         reconstruction_methods = try_multiple_normalizations(reconstructed, "Reconstruction")
        
#         # Save all methods
#         save_comparison_methods(original_seq, reconstruction_methods, output_dir, seq_idx)
        
#         # Additional debugging: check if model is actually learning
#         print(f"\n=== MODEL ANALYSIS ===")
        
#         # Check if reconstruction loss was actually decreasing
#         if 'image_loss' in metrics:
#             print(f"Final image loss: {metrics['image_loss']:.4f}")
#             if metrics['image_loss'] > 1.0:
#                 print("WARNING: High image loss suggests model didn't learn well")
        
#         # Check if the model is just outputting constant values
#         unique_vals = len(np.unique(reconstructed.numpy()))
#         print(f"Unique values in reconstruction: {unique_vals}")
#         if unique_vals < 10:
#             print("WARNING: Very few unique values - model might be outputting constants")
        
#         # Try to see if there's any structure in the reconstruction
#         recon_np = reconstructed.numpy()
#         if recon_np.std() < 0.01:
#             print("WARNING: Very low std deviation - reconstruction is nearly constant")
#         else:
#             print(f"Reconstruction has reasonable variation (std: {recon_np.std():.4f})")

#     print(f"\nDebugging complete! Check '{output_dir}' for results.")
#     print("Look at the different 'method_*.png' files to see which normalization works best.")


# def main():
#     parser = argparse.ArgumentParser(description='Advanced WorldModel debugging')
#     parser.add_argument('--config', type=str, default='worldmodel_config.yaml')
#     parser.add_argument('--hdf5', type=str, required=True)
#     parser.add_argument('--checkpoint', type=str, required=True)
#     parser.add_argument('--size', type=str, default='1m')
#     parser.add_argument('--sequence_idx', type=int, default=0)
#     parser.add_argument('--output_dir', type=str, default='./advanced_debug')
    
#     args = parser.parse_args()
    
#     advanced_worldmodel_debug(
#         config_path=args.config,
#         hdf5_path=args.hdf5,
#         checkpoint_path=args.checkpoint,
#         size=args.size,
#         output_dir=args.output_dir,
#         sequence_idx=args.sequence_idx
#     )

# if __name__ == '__main__':
#     main()

"""
Advanced WorldModel Debug Visualization
This version tries multiple approaches to get proper reconstructions
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import yaml
import argparse
from pathlib import Path
from gymnasium.spaces import Box, Dict

from training_wm_v2 import (
    Config,
    apply_size_override,
    setup_config_for_worldmodel,  
    load_checkpoint
)

from models import WorldModel
from hdf5_converter import convert_hdf5_to_dreamer


def analyze_tensor_stats(tensor, name):
    """Print detailed statistics about a tensor"""
    if torch.is_tensor(tensor):
        tensor_np = tensor.detach().cpu().numpy()
    else:
        tensor_np = tensor
    
    print(f"\n{name} Statistics:")
    print(f"  Shape: {tensor_np.shape}")
    print(f"  Range: [{tensor_np.min():.4f}, {tensor_np.max():.4f}]")
    print(f"  Mean: {tensor_np.mean():.4f}, Std: {tensor_np.std():.4f}")
    print(f"  Unique values: {len(np.unique(tensor_np))}")
    
    # Check for common ranges
    if tensor_np.min() >= 0 and tensor_np.max() <= 1:
        print("  -> Appears to be in [0,1] range")
    elif tensor_np.min() >= -1 and tensor_np.max() <= 1:
        print("  -> Appears to be in [-1,1] range")
    elif tensor_np.min() >= 0 and tensor_np.max() <= 255:
        print("  -> Appears to be in [0,255] range")
    else:
        print("  -> Custom range")


def try_multiple_normalizations(tensor, name):
    """Try different normalization approaches and return the best looking one"""
    if torch.is_tensor(tensor):
        tensor = tensor.detach().cpu().numpy()
    
    tensor = tensor.copy()
    
    print(f"\nTrying different normalizations for {name}:")
    
    # Method 1: Direct clipping to [0,1]
    method1 = np.clip(tensor, 0, 1)
    print(f"  Method 1 (clip to [0,1]): range [{method1.min():.4f}, {method1.max():.4f}]")
    
    # Method 2: Min-max normalization
    if tensor.max() != tensor.min():
        method2 = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        print(f"  Method 2 (min-max norm): range [{method2.min():.4f}, {method2.max():.4f}]")
    else:
        method2 = np.zeros_like(tensor)
        print(f"  Method 2 (min-max norm): constant tensor, using zeros")
    
    # Method 3: Sigmoid activation (if model outputs logits)
    method3 = 1 / (1 + np.exp(-tensor))  # sigmoid
    print(f"  Method 3 (sigmoid): range [{method3.min():.4f}, {method3.max():.4f}]")
    
    # Method 4: Tanh to [0,1] 
    method4 = (np.tanh(tensor) + 1) / 2
    print(f"  Method 4 (tanh->01): range [{method4.min():.4f}, {method4.max():.4f}]")
    
    # Method 5: Add offset and scale (if model learned a different range)
    method5 = tensor + 0.5  # Common if model outputs around [-0.5, 0.5]
    method5 = np.clip(method5, 0, 1)
    print(f"  Method 5 (+0.5 clip): range [{method5.min():.4f}, {method5.max():.4f}]")
    
    return {
        'clip_01': method1,
        'minmax': method2, 
        'sigmoid': method3,
        'tanh': method4,
        'offset': method5,
        'raw': tensor
    }


def save_comparison_methods(original, reconstructed_methods, output_dir, seq_idx):
    """Save comparison grids for different reconstruction methods"""
    
    # Normalize original
    if torch.is_tensor(original):
        original = original.detach().cpu().numpy()
    
    if original.max() > 1:
        original = original / 255.0
    original = np.clip(original, 0, 1)
    
    # Ensure channel order is correct
    if original.ndim == 4 and original.shape[1] == 3:
        original = np.transpose(original, (0, 2, 3, 1))
    
    # Create comparison for each method
    for method_name, reconstructed in reconstructed_methods.items():
        if reconstructed.ndim == 4 and reconstructed.shape[1] == 3:
            reconstructed = np.transpose(reconstructed, (0, 2, 3, 1))
        
        # Create grid comparing first 8 frames
        num_frames = min(8, original.shape[0])
        fig, axes = plt.subplots(2, num_frames, figsize=(num_frames*3, 6))
        
        if num_frames == 1:
            axes = axes.reshape(2, 1)
        
        fig.suptitle(f'Method: {method_name} - Original vs Reconstructed', fontsize=16)
        
        for i in range(num_frames):
            # Original
            axes[0, i].imshow(original[i])
            axes[0, i].set_title(f'Original {i}')
            axes[0, i].axis('off')
            
            # Reconstructed
            axes[1, i].imshow(reconstructed[i])
            axes[1, i].set_title(f'Recon {i}')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f'method_{method_name}_seq_{seq_idx}.png')
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {method_name} comparison: {save_path}")


def advanced_worldmodel_debug(config_path, hdf5_path, checkpoint_path, size='1m', 
                            output_dir='./advanced_debug', sequence_idx=0):
    """Advanced debugging with multiple reconstruction attempts"""
    
    print(f"=== ADVANCED WORLDMODEL DEBUG ===")
    print(f"Loading model: {checkpoint_path}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    size_key = f"size{size}"
    apply_size_override(config, size_key)
    cfg_dict = config['defaults'].copy()
    
    size_overrides = config.get(size_key, {})
    cfg_dict = setup_config_for_worldmodel(cfg_dict, size_overrides)

    # Load data
    data = convert_hdf5_to_dreamer(hdf5_path)
    analyze_tensor_stats(data['image'], "Original Data")
    
    # Store truly original images
    original_images = data['image'].clone()
    
    # Normalize for model
    if data['image'].max() > 1.0:
        data['image'] = data['image'] / 255.0
    
    # Add missing data
    if 'reward' not in data:
        data['reward'] = torch.zeros(data['action'].shape[:2], dtype=torch.float32)
    if 'discount' not in data:
        data['discount'] = torch.ones(data['action'].shape[:2], dtype=torch.float32)
    
    # Setup model
    actual_action_dim = data['action'].shape[-1]
    cfg_dict['num_actions'] = actual_action_dim
    cfg_dict['batch_size'] = 16
    cfg_dict['batch_length'] = 64

    cfg = Config(**cfg_dict)
    device = cfg_dict.get('device', 'cuda')

    image_shape = data['image'].shape[2:]
    obs_space = Dict({'image': Box(low=0, high=1, shape=image_shape, dtype=np.float32)})
    act_space = Box(low=-1.0, high=1.0, shape=(actual_action_dim,), dtype=np.float32)

    # Initialize and load model
    wm = WorldModel(obs_space, act_space, 0, cfg).to(device)
    epoch, metrics = load_checkpoint(wm, checkpoint_path)
    print(f"\nLoaded model from epoch {epoch}")
    print("Training metrics:", {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in metrics.items()})
    
    wm.eval()

    # Process sequence
    seq_idx = sequence_idx
    batch_length = min(cfg_dict['batch_length'], data['image'].shape[1])
    
    original_seq = original_images[seq_idx, :batch_length]
    analyze_tensor_stats(original_seq, "Original Sequence")
    
    # Prepare batch
    batch_data = {
        'image': data['image'][seq_idx:seq_idx+1, :batch_length].to(device),
        'action': data['action'][seq_idx:seq_idx+1, :batch_length].to(device),
        'reward': data['reward'][seq_idx:seq_idx+1, :batch_length].to(device),
        'is_first': data['is_first'][seq_idx:seq_idx+1, :batch_length].to(device),
        'is_terminal': data['is_terminal'][seq_idx:seq_idx+1, :batch_length].to(device),
        'discount': data['discount'][seq_idx:seq_idx+1, :batch_length].to(device),
    }

    print(f"\n=== MODEL FORWARD PASS ===")
    
    with torch.no_grad():
        # Step-by-step model forward pass
        print("1. Preprocessing...")
        preprocessed = wm.preprocess(batch_data)
        analyze_tensor_stats(preprocessed['image'], "Preprocessed Image")
        
        print("2. Encoding...")
        embed = wm.encoder(preprocessed)
        analyze_tensor_stats(embed, "Embeddings")
        
        print("3. Dynamics...")
        post, prior = wm.dynamics.observe(embed, preprocessed["action"], preprocessed["is_first"])
        feat = wm.dynamics.get_feat(post)
        analyze_tensor_stats(feat, "Features")
        
        print("4. Decoding...")
        decoder_pred = wm.heads["decoder"](feat)
        recon_dist = decoder_pred["image"]
        
        # Handle different distribution types
        if hasattr(recon_dist, 'mode'):
            reconstructed = recon_dist.mode()
            print("Using distribution mode")
        elif hasattr(recon_dist, 'mean'):
            reconstructed = recon_dist.mean()
            print("Using distribution mean")
        elif hasattr(recon_dist, 'sample'):
            reconstructed = recon_dist.sample()
            print("Using distribution sample")
        else:
            reconstructed = recon_dist
            print("Using raw output")
        
        analyze_tensor_stats(reconstructed, "Raw Reconstruction")
        
        # Check what type of distribution this is
        print(f"Decoder output type: {type(recon_dist)}")
        if hasattr(recon_dist, '__dict__'):
            print(f"Distribution attributes: {list(recon_dist.__dict__.keys())}")
        
        # Remove batch dimension
        reconstructed = reconstructed[0].cpu()  # [T, H, W, C]
        
        print(f"\n=== TRYING DIFFERENT NORMALIZATIONS ===")
        reconstruction_methods = try_multiple_normalizations(reconstructed, "Reconstruction")
        
        # Save all methods
        save_comparison_methods(original_seq, reconstruction_methods, output_dir, seq_idx)
        
        # Additional debugging: check if model is actually learning
        print(f"\n=== MODEL ANALYSIS ===")
        
        # Check if reconstruction loss was actually decreasing
        if 'image_loss' in metrics:
            print(f"Final image loss: {metrics['image_loss']:.4f}")
            if metrics['image_loss'] > 1.0:
                print("WARNING: High image loss suggests model didn't learn well")
        
        # Check if the model is just outputting constant values
        unique_vals = len(np.unique(reconstructed.numpy()))
        print(f"Unique values in reconstruction: {unique_vals}")
        if unique_vals < 10:
            print("WARNING: Very few unique values - model might be outputting constants")
        
        # Try to see if there's any structure in the reconstruction
        recon_np = reconstructed.numpy()
        if recon_np.std() < 0.01:
            print("WARNING: Very low std deviation - reconstruction is nearly constant")
        else:
            print(f"Reconstruction has reasonable variation (std: {recon_np.std():.4f})")

    print(f"\nDebugging complete! Check '{output_dir}' for results.")
    print("Look at the different 'method_*.png' files to see which normalization works best.")


def main():
    parser = argparse.ArgumentParser(description='Advanced WorldModel debugging')
    parser.add_argument('--config', type=str, default='worldmodel_config.yaml')
    parser.add_argument('--hdf5', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--size', type=str, default='1m')
    parser.add_argument('--sequence_idx', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='./advanced_debug')
    
    args = parser.parse_args()
    
    advanced_worldmodel_debug(
        config_path=args.config,
        hdf5_path=args.hdf5,
        checkpoint_path=args.checkpoint,
        size=args.size,
        output_dir=args.output_dir,
        sequence_idx=args.sequence_idx
    )

if __name__ == '__main__':
    main()