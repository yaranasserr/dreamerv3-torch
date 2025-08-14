"""
# Visualize single file results
python visualizations.py \
  --hdf5 robot_data.hdf5 \
  --checkpoint ./checkpoints/best_model.pt \
  --config worldmodel_config.yaml \
  --size 1m \
  --num_sequences 5 \
  --output_dir ./results
```


### Skip videos, only create comparison grids
```
python visualizations.py --hdf5 data.hdf5 --checkpoint model.pt --no_videos
```
### Skip grids, only create videos  
```
python visualizations.py --hdf5 data.hdf5 --checkpoint model.pt --no_grids
```
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


def normalize_image(img):
    """Normalize image tensor to [0, 1] range for visualization"""
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()
    
    if img.max() > 1.0:
        img = img / 255.0
    
    img = np.clip(img, 0, 1)
    return img


def prepare_video_comparison(original, reconstructed):
    """Prepare original and reconstructed videos for side-by-side comparison"""
    if torch.is_tensor(original):
        original = original.detach().cpu().numpy()
    if torch.is_tensor(reconstructed):
        reconstructed = reconstructed.detach().cpu().numpy()
    
    original = normalize_image(original)
    reconstructed = normalize_image(reconstructed)
    
    if original.ndim == 4:
        if original.shape[1] == 3:
            original = np.transpose(original, (0, 2, 3, 1))
        elif original.shape[-1] != 3:
            if original.shape[-1] == 1:
                original = np.repeat(original, 3, axis=-1)
    
    if reconstructed.ndim == 4:
        if reconstructed.shape[1] == 3:
            reconstructed = np.transpose(reconstructed, (0, 2, 3, 1))
        elif reconstructed.shape[-1] != 3:
            if reconstructed.shape[-1] == 1:
                reconstructed = np.repeat(reconstructed, 3, axis=-1)
    
    return original, reconstructed


def create_comparison_grid(original, reconstructed, num_frames=16):
    """Create a grid showing original vs reconstructed frames"""
    original, reconstructed = prepare_video_comparison(original, reconstructed)
    
    total_frames = min(original.shape[0], reconstructed.shape[0])
    frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    
    rows = 4
    cols = num_frames // rows
    fig, axes = plt.subplots(2*rows, cols, figsize=(cols*3, rows*6))
    fig.suptitle('Original vs Reconstructed Frames', fontsize=16)
    
    for i, frame_idx in enumerate(frame_indices):
        row = (i // cols) * 2
        col = i % cols
        
        axes[row, col].imshow(original[frame_idx])
        axes[row, col].set_title(f'Original Frame {frame_idx}', fontsize=10)
        axes[row, col].axis('off')
        
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
    
    if frames.ndim == 4:
        if frames.shape[1] == 3:
            frames = np.transpose(frames, (0, 2, 3, 1))
    
    frames = (frames * 255).astype(np.uint8)
    T, H, W, C = frames.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    
    for frame in frames:
        if C == 3:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame
        out.write(frame_bgr)
    
    out.release()
    print(f"Video saved to: {output_path}")


def visualize_worldmodel_reconstruction(config_path, hdf5_path, checkpoint_path, size='1m', 
                                      output_dir='./visualizations', sequence_idx=0, 
                                      num_sequences=1, save_videos=True, save_grids=True):
    """Visualize WorldModel reconstructions using trained model"""
    
    print(f"Loading trained model from: {checkpoint_path}")
    
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

    print("Loading data...")
    data = convert_hdf5_to_dreamer(hdf5_path)
    
    
    actual_action_dim = data['action'].shape[-1]
    cfg_dict['num_actions'] = actual_action_dim
    cfg_dict['batch_size'] = 16
    cfg_dict['batch_length'] = 64

    cfg = Config(**cfg_dict)
    device = cfg_dict.get('device', 'cuda')

    image_shape = data['image'].shape[2:]
    
    obs_space = Dict({
        'image': Box(low=0, high=1, shape=image_shape, dtype=np.float32)
    })
    act_space = Box(low=-1.0, high=1.0, shape=(actual_action_dim,), dtype=np.float32)

    # Initialize and load trained model
    wm = WorldModel(obs_space, act_space, 0, cfg).to(device)
    epoch, metrics = load_checkpoint(wm, checkpoint_path)
    print(f"Loaded model from epoch {epoch}")
    print(f"Model metrics: {metrics}")
    
    wm.eval()

    # Process sequences
    for seq_idx in range(sequence_idx, min(sequence_idx + num_sequences, data['image'].shape[0])):
        print(f"\nProcessing sequence {seq_idx}...")
        
        batch_length = min(cfg_dict['batch_length'], data['image'].shape[1])
        
        batch_data = {
            'image': data['image'][seq_idx:seq_idx+1, :batch_length].to(device),
            'action': data['action'][seq_idx:seq_idx+1, :batch_length].to(device),
            'is_first': data['is_first'][seq_idx:seq_idx+1, :batch_length].to(device),
            'is_terminal': data['is_terminal'][seq_idx:seq_idx+1, :batch_length].to(device),
        }

        try:
            with torch.no_grad():
                video_pred = wm.video_pred(batch_data)
            
            # Extract frames - video_pred contains [truth, model, error] concatenated
            video_pred = video_pred[0]  # Remove batch dimension
            H = video_pred.shape[1] // 3  # Each section is 1/3 of height
            
            original_video = video_pred[:, :H]  # Top third
            reconstructed_video = video_pred[:, H:2*H]  # Middle third
            
            print(f"Original video shape: {original_video.shape}")
            print(f"Reconstructed video shape: {reconstructed_video.shape}")
            
            if save_grids:
                fig = create_comparison_grid(original_video, reconstructed_video)
                grid_path = os.path.join(output_dir, f'comparison_grid_seq_{seq_idx}.png')
                fig.savefig(grid_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"Comparison grid saved to: {grid_path}")
            
            if save_videos:
                orig_path = os.path.join(output_dir, f'original_seq_{seq_idx}.mp4')
                save_video_mp4(original_video, orig_path)
                
                recon_path = os.path.join(output_dir, f'reconstructed_seq_{seq_idx}.mp4')
                save_video_mp4(reconstructed_video, recon_path)
            
        except Exception as e:
            print(f"Error processing sequence {seq_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\nVisualization complete! Check the '{output_dir}' directory for results.")


def main():
    parser = argparse.ArgumentParser(description='Visualize WorldModel reconstructions')
    parser.add_argument('--config', type=str, default='worldmodel_config.yaml',
                       help='Path to config YAML file')
    parser.add_argument('--hdf5', type=str, required=True,
                       help='Path to HDF5 file for visualization')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--size', type=str, default='1m',
                       help='Model size (1m, 12m, 70m)')
    parser.add_argument('--sequence_idx', type=int, default=0,
                       help='Starting sequence index')
    parser.add_argument('--num_sequences', type=int, default=5,
                       help='Number of sequences to visualize')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for visualizations')
    parser.add_argument('--no_videos', action='store_true',
                       help='Skip saving videos (only save grids)')
    parser.add_argument('--no_grids', action='store_true',
                       help='Skip saving grids (only save videos)')
    
    args = parser.parse_args()
    
    visualize_worldmodel_reconstruction(
        config_path=args.config,
        hdf5_path=args.hdf5,
        checkpoint_path=args.checkpoint,
        size=args.size,
        output_dir=args.output_dir,
        sequence_idx=args.sequence_idx,
        num_sequences=args.num_sequences,
        save_videos=not args.no_videos,
        save_grids=not args.no_grids
    )


if __name__ == '__main__':
    main()