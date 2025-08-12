import argparse
import torch
import yaml
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import cv2
import os
from pathlib import Path
import gymnasium as gym
from gymnasium.spaces import Box, Dict
from models import WorldModel
from hdf5_converter import convert_hdf5_to_dreamer
from types import SimpleNamespace


class Config:
    """Config class that supports both attribute access and dictionary unpacking"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                setattr(self, key, Config(**value))
            else:
                setattr(self, key, value)
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __contains__(self, key):
        return hasattr(self, key)
    
    def get(self, key, default=None):
        return getattr(self, key, default)
    
    def keys(self):
        return vars(self).keys()
    
    def items(self):
        for key, value in vars(self).items():
            if isinstance(value, Config):
                yield key, dict(value.items())
            else:
                yield key, value
    
    def to_dict(self):
        result = {}
        for key, value in vars(self).items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


def load_config(config_path, config_name='defaults'):
    """Load configuration from YAML file, supporting both repo format and custom configs"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Debug - Loaded config keys: {list(config.keys()) if isinstance(config, dict) else 'Not a dict'}")
    
    # Handle different config formats
    if isinstance(config, dict):
        # If it's the repo format with named configurations
        if config_name in config and isinstance(config[config_name], dict):
            if config_name == 'defaults':
                return config['defaults'].copy()
            else:
                # Merge the specific config with defaults
                base_config = config.get('defaults', {}).copy()
                specific_config = config[config_name]
                for key, value in specific_config.items():
                    if isinstance(value, dict) and key in base_config and isinstance(base_config[key], dict):
                        base_config[key].update(value)
                    else:
                        base_config[key] = value
                return base_config
        
        # If config_name is 'defaults' but not found, try to use the whole config
        elif config_name == 'defaults':
            return config
        
        # If specific config_name not found, try to use 'defaults' section
        elif 'defaults' in config:
            print(f"Warning: {config_name} not found, using 'defaults'")
            return config['defaults']
        
        # Last resort: use the whole config as-is
        else:
            print(f"Warning: No 'defaults' section found, using entire config")
            return config
    else:
        print(f"Error: Config file is not a valid dictionary")
        return {}


def setup_config_for_worldmodel(cfg_dict, config_name='defaults'):
    """Setup config with all required parameters for WorldModel using repo format"""
    
    # Handle different config formats - ensure cfg_dict is a dictionary
    if not isinstance(cfg_dict, dict):
        print(f"Warning: Config is not a dictionary, got {type(cfg_dict)}")
        cfg_dict = {}
    
    # The repo config already has the right structure, just ensure all required keys exist
    required_defaults = {
        'device': 'cuda:0',
        'batch_size': 16,
        'batch_length': 64,
        'dyn_hidden': 512,
        'dyn_deter': 512,
        'dyn_stoch': 32,
        'dyn_discrete': 32,
        'dyn_rec_depth': 1,
        'dyn_mean_act': 'none',
        'dyn_std_act': 'sigmoid2',
        'dyn_min_std': 0.1,
        'unimix_ratio': 0.01,
        'initial': 'learned',
        'kl_free': 1.0,
        'dyn_scale': 0.5,
        'rep_scale': 0.1,
        'act': 'SiLU',
        'norm': True,
        'units': 512,
        'grad_heads': ['decoder', 'reward', 'cont'],
        
        # Encoder defaults (repo format)
        'encoder': {
            'mlp_keys': '$^',
            'cnn_keys': 'image',
            'act': 'SiLU',
            'norm': True,
            'cnn_depth': 32,
            'kernel_size': 4,
            'minres': 4,
            'mlp_layers': 5,
            'mlp_units': 1024,
            'symlog_inputs': True
        },
        
        # Decoder defaults (repo format)
        'decoder': {
            'mlp_keys': '$^',
            'cnn_keys': 'image',
            'act': 'SiLU',
            'norm': True,
            'cnn_depth': 32,
            'kernel_size': 4,
            'minres': 4,
            'mlp_layers': 5,
            'mlp_units': 1024,
            'cnn_sigmoid': False,
            'image_dist': 'mse',
            'vector_dist': 'symlog_mse',
            'outscale': 1.0
        },
        
        # Head configurations (repo format)
        'reward_head': {
            'layers': 2,
            'dist': 'symlog_disc',
            'loss_scale': 1.0,
            'outscale': 0.0
        },
        
        'cont_head': {
            'layers': 2,
            'loss_scale': 1.0,
            'outscale': 1.0
        }
    }
    
    # Apply defaults only for missing keys
    for key, value in required_defaults.items():
        if key not in cfg_dict:
            cfg_dict[key] = value
        elif key in ['encoder', 'decoder', 'reward_head', 'cont_head'] and isinstance(cfg_dict[key], dict):
            # Merge nested dictionaries, preserving existing values
            for subkey, subvalue in value.items():
                if subkey not in cfg_dict[key]:
                    cfg_dict[key][subkey] = subvalue
    
    # Debug: Print key dimensions to verify
    print(f"Debug - Model Config:")
    print(f"  batch_size: {cfg_dict.get('batch_size', 'NOT SET')}")
    print(f"  batch_length: {cfg_dict.get('batch_length', 'NOT SET')}")
    print(f"  dyn_stoch: {cfg_dict.get('dyn_stoch', 'NOT SET')}")
    print(f"  dyn_discrete: {cfg_dict.get('dyn_discrete', 'NOT SET')}")
    print(f"  dyn_deter: {cfg_dict.get('dyn_deter', 'NOT SET')}")
    print(f"  dyn_hidden: {cfg_dict.get('dyn_hidden', 'NOT SET')}")
    print(f"  encoder.cnn_depth: {cfg_dict.get('encoder', {}).get('cnn_depth', 'NOT SET')}")
    print(f"  decoder.cnn_depth: {cfg_dict.get('decoder', {}).get('cnn_depth', 'NOT SET')}")
    
    return cfg_dict


def normalize_image(img):
    """Normalize image tensor to [0, 1] range for visualization"""
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()
    
    # Handle different input ranges
    if img.max() > 1.0:  # Assume [0, 255] range
        img = img / 255.0
    
    # Clip to valid range
    img = np.clip(img, 0, 1)
    return img


def prepare_video_comparison(original, reconstructed):
    """Prepare original and reconstructed videos for side-by-side comparison"""
    # Ensure both are numpy arrays and normalized
    if torch.is_tensor(original):
        original = original.detach().cpu().numpy()
    if torch.is_tensor(reconstructed):
        reconstructed = reconstructed.detach().cpu().numpy()
    
    original = normalize_image(original)
    reconstructed = normalize_image(reconstructed)
    
    # Handle channel dimension - convert to (T, H, W, C) if needed
    if original.ndim == 4:
        if original.shape[1] == 3:  # (T, C, H, W) -> (T, H, W, C)
            original = np.transpose(original, (0, 2, 3, 1))
        elif original.shape[-1] != 3:  # (T, H, W, C) but C != 3
            if original.shape[-1] == 1:  # Grayscale
                original = np.repeat(original, 3, axis=-1)
    
    if reconstructed.ndim == 4:
        if reconstructed.shape[1] == 3:  # (T, C, H, W) -> (T, H, W, C)
            reconstructed = np.transpose(reconstructed, (0, 2, 3, 1))
        elif reconstructed.shape[-1] != 3:  # (T, H, W, C) but C != 3
            if reconstructed.shape[-1] == 1:  # Grayscale
                reconstructed = np.repeat(reconstructed, 3, axis=-1)
    
    return original, reconstructed


def create_comparison_grid(original, reconstructed, num_frames=16):
    """Create a grid showing original vs reconstructed frames"""
    original, reconstructed = prepare_video_comparison(original, reconstructed)
    
    # Select frames to display
    total_frames = min(original.shape[0], reconstructed.shape[0])
    frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    
    # Create figure
    rows = 4
    cols = num_frames // rows
    fig, axes = plt.subplots(2*rows, cols, figsize=(cols*3, rows*6))
    fig.suptitle('Original vs Reconstructed Frames', fontsize=16)
    
    for i, frame_idx in enumerate(frame_indices):
        row = (i // cols) * 2
        col = i % cols
        
        # Original frame
        axes[row, col].imshow(original[frame_idx])
        axes[row, col].set_title(f'Original Frame {frame_idx}', fontsize=10)
        axes[row, col].axis('off')
        
        # Reconstructed frame
        axes[row+1, col].imshow(reconstructed[frame_idx])
        axes[row+1, col].set_title(f'Reconstructed Frame {frame_idx}', fontsize=10)
        axes[row+1, col].axis('off')
    
    plt.tight_layout()
    return fig


def save_video_mp4(frames, output_path, fps=10):
    """Save frames as MP4 video"""
    if torch.is_tensor(frames):
        frames = frames.detach().cpu().numpy()
    
    frames = normalize_image(frames)
    
    # Handle channel dimension
    if frames.ndim == 4:
        if frames.shape[1] == 3:  # (T, C, H, W) -> (T, H, W, C)
            frames = np.transpose(frames, (0, 2, 3, 1))
    
    # Convert to uint8
    frames = (frames * 255).astype(np.uint8)
    
    # Get video dimensions
    T, H, W, C = frames.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    
    for frame in frames:
        if C == 3:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame
        out.write(frame_bgr)
    
    out.release()
    print(f"Video saved to: {output_path}")


def create_side_by_side_video(original, reconstructed, output_path, fps=10):
    """Create side-by-side comparison video"""
    original, reconstructed = prepare_video_comparison(original, reconstructed)
    
    # Convert to uint8
    original = (original * 255).astype(np.uint8)
    reconstructed = (reconstructed * 255).astype(np.uint8)
    
    # Get dimensions
    T, H, W, C = original.shape
    
    # Create side-by-side frames
    combined_frames = np.zeros((T, H, W*2 + 10, C), dtype=np.uint8)  # 10 pixel gap
    combined_frames[:, :, :W] = original
    combined_frames[:, :, W+10:] = reconstructed
    
    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    color = (255, 255, 255)
    thickness = 2
    
    for i in range(T):
        cv2.putText(combined_frames[i], 'Original', (10, 30), font, font_scale, color, thickness)
        cv2.putText(combined_frames[i], 'Reconstructed', (W+20, 30), font, font_scale, color, thickness)
    
    # Save video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (W*2 + 10, H))
    
    for frame in combined_frames:
        if C == 3:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame
        out.write(frame_bgr)
    
    out.release()
    print(f"Side-by-side video saved to: {output_path}")


def visualize_worldmodel_reconstruction(config_path, hdf5_path, config_name='defaults', output_dir='./visualizations', 
                                      sequence_idx=0, num_sequences=1, save_videos=True, save_grids=True):
    """Main function to visualize WorldModel reconstructions"""
    
    print(f"Visualizing WorldModel reconstructions with config: {config_name}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load config using repo format
    cfg_dict = load_config(config_path, config_name)
    print(f"Loaded config: {config_name}")
    
    # Setup config for WorldModel
    cfg_dict = setup_config_for_worldmodel(cfg_dict, config_name)

    print("Loading data...")
    data = convert_hdf5_to_dreamer(hdf5_path)
    
    actual_action_dim = data['action'].shape[-1]
    cfg_dict['num_actions'] = actual_action_dim
    print(f"Action dimension from data: {actual_action_dim}")

    cfg = Config(**cfg_dict)
    device = cfg_dict.get('device', 'cuda:0')
    batch_size = cfg_dict['batch_size']
    batch_length = cfg_dict['batch_length']
    num_actions = cfg_dict['num_actions']

    image_shape = data['image'].shape[2:]
    print(f"Image shape: {image_shape}")

    obs_space = Dict({
        'image': Box(low=0, high=255, shape=image_shape, dtype=np.uint8)
    })
    act_space = Box(low=-1.0, high=1.0, shape=(num_actions,), dtype=np.float32)

    print("Initializing WorldModel...")
    wm = WorldModel(obs_space, act_space, 0, cfg).to(device)
    print("WorldModel initialized successfully!")

    # Process multiple sequences
    for seq_idx in range(sequence_idx, min(sequence_idx + num_sequences, data['image'].shape[0])):
        print(f"\nProcessing sequence {seq_idx}...")
        
        # Prepare batch data for single sequence
        batch_data = {
            'image': data['image'][seq_idx:seq_idx+1, :batch_length].to(device),
            'action': data['action'][seq_idx:seq_idx+1, :batch_length].to(device),
            'reward': data['reward'][seq_idx:seq_idx+1, :batch_length].to(device),
            'is_first': data['is_first'][seq_idx:seq_idx+1, :batch_length].to(device),
            'is_terminal': data['is_terminal'][seq_idx:seq_idx+1, :batch_length].to(device),
            'discount': data['discount'][seq_idx:seq_idx+1, :batch_length].to(device),
        }

        try:
            # Get video prediction
            with torch.no_grad():
                video_pred = wm.video_pred(batch_data)
            
            print(f"Video prediction shape: {video_pred.shape}")
            
            # Extract original and reconstructed videos
            original_video = batch_data['image'][0]  # Remove batch dimension
            reconstructed_video = video_pred[0]  # Remove batch dimension
            
            print(f"Original video shape: {original_video.shape}")
            print(f"Reconstructed video shape: {reconstructed_video.shape}")
            
            # Save comparison grid
            if save_grids:
                fig = create_comparison_grid(original_video, reconstructed_video)
                grid_path = os.path.join(output_dir, f'comparison_grid_seq_{seq_idx}_{config_name}.png')
                fig.savefig(grid_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"Comparison grid saved to: {grid_path}")
            
            # Save videos
            if save_videos:
                # Save original video
                orig_path = os.path.join(output_dir, f'original_seq_{seq_idx}_{config_name}.mp4')
                save_video_mp4(original_video, orig_path)
                
                # Save reconstructed video
                recon_path = os.path.join(output_dir, f'reconstructed_seq_{seq_idx}_{config_name}.mp4')
                save_video_mp4(reconstructed_video, recon_path)
                
                # Save side-by-side comparison
                comparison_path = os.path.join(output_dir, f'comparison_seq_{seq_idx}_{config_name}.mp4')
                create_side_by_side_video(original_video, reconstructed_video, comparison_path)
            
        except Exception as e:
            print(f"Error processing sequence {seq_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\nVisualization complete! Check the '{output_dir}' directory for results.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize DreamerV3 WorldModel reconstructions')
    parser.add_argument('--config', type=str, default='configs/defaults.yaml',
                       help='Path to config YAML file')
    parser.add_argument('--hdf5', type=str, required=True,
                       help='Path to HDF5 data file')
    parser.add_argument('--config_name', type=str, default='defaults',
                       help='Config name to use (defaults, dmc_vision, crafter, atari100k, etc.)')
    parser.add_argument('--output_dir', type=str, default='./visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--sequence_idx', type=int, default=0,
                       help='Starting sequence index to visualize')
    parser.add_argument('--num_sequences', type=int, default=1,
                       help='Number of sequences to visualize')
    parser.add_argument('--no_videos', action='store_true',
                       help='Skip saving videos (only save grids)')
    parser.add_argument('--no_grids', action='store_true',
                       help='Skip saving grids (only save videos)')
    
    args = parser.parse_args()
    
    visualize_worldmodel_reconstruction(
        config_path=args.config,
        hdf5_path=args.hdf5,
        config_name=args.config_name,
        output_dir=args.output_dir,
        sequence_idx=args.sequence_idx,
        num_sequences=args.num_sequences,
        save_videos=not args.no_videos,
        save_grids=not args.no_grids
    )