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
import time
import json
from torch.utils.tensorboard import SummaryWriter
import pickle
from datetime import datetime
import glob


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


def apply_size_override(config, size_key):
    """Apply overrides like 'size12m' to base config."""
    if size_key in config:
        overrides = config[size_key]
        for pattern, values in overrides.items():
            for conf_key in list(config['defaults'].keys()):
                if re.fullmatch(pattern, conf_key):
                    if isinstance(config['defaults'][conf_key], dict) and isinstance(values, dict):
                        config['defaults'][conf_key].update(values)
                    else:
                        config['defaults'][conf_key] = values
            
            if pattern not in config['defaults']:
                config['defaults'][pattern] = values


def setup_config_for_worldmodel(cfg_dict, size_overrides):
    """Setup configuration for WorldModel with proper heads for image-only training"""
    rssm_key = '.*\\.rssm'
    depth_key = '.*\\.depth'
    units_key = '.*\\.units'
    
    for pattern, values in size_overrides.items():
        if pattern == rssm_key:
            cfg_dict.update({
                'dyn_stoch': values.get('classes', 32),
                'dyn_deter': values.get('deter', 512), 
                'dyn_hidden': values.get('hidden', 512),
                'dyn_discrete': values.get('discrete', 32),
            })
        elif pattern == depth_key:
            cfg_dict['encoder'] = cfg_dict.get('encoder', {})
            cfg_dict['decoder'] = cfg_dict.get('decoder', {})
            cfg_dict['encoder']['cnn_depth'] = values
            cfg_dict['decoder']['cnn_depth'] = values
        elif pattern == units_key:
            cfg_dict['units'] = values
            cfg_dict['encoder'] = cfg_dict.get('encoder', {})
            cfg_dict['decoder'] = cfg_dict.get('decoder', {})
            cfg_dict['encoder']['mlp_units'] = values
            cfg_dict['decoder']['mlp_units'] = values
    
    required_defaults = {
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
        'grad_heads': ['decoder'],  # Only train decoder, disable continuation head
        
        # Training specific parameters
        'model_lr': 1e-4,
        'opt_eps': 1e-8,
        'grad_clip': 1000.0,
        'weight_decay': 0.0,
        'opt': 'adam',
        'precision': 32,
        'discount': 0.99,  # Keep for compatibility
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        
        'encoder': {
            'mlp_keys': '$^',  # No MLP inputs - this regex matches nothing
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
        
  
        'reward_head': {
            'layers': 2,
            'dist': 'symlog_disc',
            'loss_scale': 0.0,  # Disabled
            'outscale': 0.0
        },
        
        'cont_head': {
            'layers': 2,
            'dist': 'symlog_disc', 
            'loss_scale': 0.0, 
            'outscale': 0.0
        }
    }

    for key, value in required_defaults.items():
        if key not in cfg_dict:
            cfg_dict[key] = value
        elif key in ['encoder', 'decoder'] and isinstance(cfg_dict[key], dict):
            for subkey, subvalue in value.items():
                if subkey not in cfg_dict[key]:
                    cfg_dict[key][subkey] = subvalue
    
    return cfg_dict



def save_checkpoint(model, epoch, metrics, base_checkpoint_dir, is_best=False):
    """Save model checkpoints into a timestamped subfolder."""
    # Create timestamped subfolder once
    if not hasattr(save_checkpoint, "timestamp_dir"):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        save_checkpoint.timestamp_dir = os.path.join(base_checkpoint_dir, timestamp)
        os.makedirs(save_checkpoint.timestamp_dir, exist_ok=True)
    
    checkpoint_dir = save_checkpoint.timestamp_dir

    # Save checkpoint data
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': model._model_opt._opt.state_dict(),
        'metrics': metrics,
        'config': model._config.to_dict()
    }

    # Save checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)

    # Save best model
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pt')
        torch.save(checkpoint, best_path)
        print(f"New best model saved at epoch {epoch}")

    return checkpoint_path


def load_checkpoint(model, checkpoint_path):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=model._config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model._model_opt._opt.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['metrics']


def create_data_loader(data, batch_size, batch_length, shuffle=True):
    """Create data loader for training"""
    num_sequences, seq_length = data['image'].shape[:2]
    
    # Create indices for all possible subsequences
    indices = []
    for seq_idx in range(num_sequences):
        for start_idx in range(0, seq_length - batch_length + 1, batch_length // 2):
            indices.append((seq_idx, start_idx))
    
    if shuffle:
        np.random.shuffle(indices)
    
    # Create batches
    num_batches = len(indices) // batch_size
    for batch_idx in range(num_batches):
        batch_indices = indices[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        
        batch_data = {}
        for key in data.keys():
            batch_data[key] = torch.stack([
                data[key][seq_idx, start_idx:start_idx + batch_length]
                for seq_idx, start_idx in batch_indices
            ])
        
        yield batch_data


def train_worldmodel(config_path, hdf5_input, size='1m', epochs=100, batch_size=16, 
                    batch_length=64, checkpoint_dir='./checkpoints', log_dir='./logs',
                    resume_from=None, save_every=10, eval_every=5, 
                    gradient_accumulation_steps=1, max_sequences_per_file=None,
                    shuffle_files=True, sequence_mode='pad', max_sequence_length=None):
    """Train the WorldModel on multiple HDF5 files with improved error handling."""
    print(f"Starting WorldModel training with size: {size}")
    print(f"Training for {epochs} epochs")
    print(f"Batch size: {batch_size}, Batch length: {batch_length}")
    print(f"Sequence handling mode: {sequence_mode}")
    
    # Setup CUDA optimizations
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        torch.cuda.empty_cache()

    # Adjust batch size for limited GPU memory
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_memory_gb < 6:
            batch_size = min(batch_size, 8)
            batch_length = min(batch_length, 32)
            print(f"Reduced batch size to {batch_size} and length to {batch_length} for limited GPU memory")

    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Setup logging
    writer = SummaryWriter(log_dir)

    # Load and setup config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    size_key = f"size{size}"
    apply_size_override(config, size_key)
    cfg_dict = config['defaults'].copy()
    cfg_dict.update({'batch_size': batch_size, 'batch_length': batch_length})
    size_overrides = config.get(size_key, {})
    cfg_dict = setup_config_for_worldmodel(cfg_dict, size_overrides)

    # Get HDF5 files
    hdf5_files = get_hdf5_files_from_pattern(hdf5_input)
    if not hdf5_files:
        raise ValueError(f"No HDF5 files found with pattern: {hdf5_input}")
    if shuffle_files:
        import random
        random.shuffle(hdf5_files)

    print("Loading data...")
    if sequence_mode == 'pad':
        data = load_multiple_hdf5_files(
            hdf5_files, max_sequences_per_file=max_sequences_per_file,
            pad_sequences=True, max_length=max_sequence_length
        )
    elif sequence_mode == 'truncate':
        data = load_multiple_hdf5_files_truncate(
            hdf5_files, max_sequences_per_file=max_sequences_per_file,
            target_length=max_sequence_length
        )
    else:
        raise ValueError(f"Invalid sequence_mode: {sequence_mode}")

    # Data validation and preprocessing
    print("\nValidating and preprocessing data...")
    print(f"Image dtype: {data['image'].dtype}, range: [{data['image'].min():.3f}, {data['image'].max():.3f}]")
    print(f"Action dtype: {data['action'].dtype}, range: [{data['action'].min():.3f}, {data['action'].max():.3f}]")

    # Normalize images to [0, 1] if needed
    if data['image'].max() > 1.0:
        print("Normalizing images from [0, 255] to [0, 1]")
        data['image'] = data['image'] / 255.0
        print(f"After normalization - Image range: [{data['image'].min():.3f}, {data['image'].max():.3f}]")

    # Ensure correct data types
    data['image'] = data['image'].float()
    data['action'] = data['action'].float()
    data['is_terminal'] = data['is_terminal'].bool()
    data['is_first'] = data['is_first'].bool()

    # Setup model configuration
    actual_action_dim = data['action'].shape[-1]
    cfg_dict['num_actions'] = actual_action_dim
    cfg = Config(**cfg_dict)
    device = cfg_dict.get('device', 'cuda')

    # Create observation and action spaces
    image_shape = data['image'].shape[2:]  # (H, W, C)
    obs_space = Dict({'image': Box(low=0, high=1, shape=image_shape, dtype=np.float32)})
    act_space = Box(low=-1.0, high=1.0, shape=(actual_action_dim,), dtype=np.float32)

    print("Initializing WorldModel...")
    try:
        wm = WorldModel(obs_space, act_space, 0, cfg).to(device)
        print("WorldModel initialized successfully!")
    except Exception as e:
        print(f"Error initializing WorldModel: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Setup training state
    start_epoch = 0
    best_loss = float('inf')
    
    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")
        start_epoch, last_metrics = load_checkpoint(wm, resume_from)
        best_loss = last_metrics.get('model_loss', float('inf'))
        start_epoch += 1

    # Split data into train/validation
    val_split = 0.1
    num_sequences = data['image'].shape[0]
    val_size = int(num_sequences * val_split)
    indices = torch.randperm(num_sequences)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    # Create train/val splits (no discount field)
    data_keys = ['image', 'action', 'is_terminal', 'is_first']
    train_data = {k: data[k][train_indices] for k in data_keys}
    val_data = {k: data[k][val_indices] for k in data_keys}

    print(f"Training set: {len(train_indices)} sequences")
    print(f"Validation set: {len(val_indices)} sequences")

    # Training loop
    for epoch in range(start_epoch, epochs):
        print(f"\n=== Epoch {epoch + 1}/{epochs} ===")
        wm.train()
        torch.cuda.empty_cache()
        
        epoch_metrics = {}
        num_batches = 0
        successful_batches = 0

        # Training batches
        for batch in create_data_loader(train_data, batch_size, batch_length, shuffle=True):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Add dummy reward tensor (required by WorldModel but won't be trained)
            batch['reward'] = torch.zeros(batch['action'].shape[0], batch['action'].shape[1], 
                              device=device, dtype=torch.float32)
            
            # Debug info for first few batches
            if num_batches < 3:
                print(f"\nBatch {num_batches} debug info:")
                for k, v in batch.items():
                    print(f"  {k}: shape={v.shape}, dtype={v.dtype}, device={v.device}")
                    if k == 'image':
                        print(f"    image range: [{v.min().item():.3f}, {v.max().item():.3f}]")
            
            try:
                # Test preprocessing first
                if num_batches < 3:
                    print(f"  Testing preprocessing...")
                    data_processed = wm.preprocess(batch)
                    print(f"  Preprocessing successful")
                    
                    print(f"  Testing encoder...")
                    embed = wm.encoder(data_processed)
                    print(f"  Encoder successful, embed shape: {embed.shape}")
                
                # Full training step
                post, context, metrics = wm._train(batch)
                
                # Filter out reward-related metrics
                metrics = {k: v for k, v in metrics.items() 
                          if not any(x in k.lower() for x in ['reward', 'rew'])}
                
                # Accumulate metrics
                for k, v in metrics.items():
                    if k not in epoch_metrics:
                        epoch_metrics[k] = []
                    try:
                        if torch.is_tensor(v):
                            if v.numel() == 1:
                                epoch_metrics[k].append(v.item())
                            else:
                                epoch_metrics[k].append(v.mean().item())
                        elif isinstance(v, (float, int)):
                            epoch_metrics[k].append(float(v))
                        elif isinstance(v, np.ndarray):
                            epoch_metrics[k].append(float(np.mean(v)))
                    except Exception as e:
                        print(f"Warning: Could not process metric {k}: {e}")

                successful_batches += 1
                
                if num_batches % 10 == 0:
                    current_loss = epoch_metrics.get('model_loss', [0])[-1] if epoch_metrics.get('model_loss') else 0
                    print(f"Batch {num_batches}, Loss: {current_loss:.4f}, Success rate: {successful_batches}/{num_batches+1}")
                    
            except torch.cuda.OutOfMemoryError:
                print(f"OOM at batch {num_batches}. Clearing cache and skipping.")
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                print(f"Detailed error in batch {num_batches}:")
                print(f"  Error type: {type(e).__name__}")
                print(f"  Error message: {str(e)}")
                if num_batches < 5:  # Print full traceback for first few errors
                    import traceback
                    traceback.print_exc()
                torch.cuda.empty_cache()
                continue
            
            num_batches += 1
            
            # Periodic cleanup
            if num_batches % 5 == 0:
                torch.cuda.empty_cache()

        # Calculate epoch metrics
        if epoch_metrics:
            avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
            total_loss = avg_metrics.get('model_loss', float('inf'))
            print(f"Train loss: {total_loss:.4f} (from {successful_batches}/{num_batches} successful batches)")

            # Log training metrics
            for k, v in avg_metrics.items():
                writer.add_scalar(f'train/{k}', v, epoch)
        else:
            print("No successful training batches this epoch!")
            total_loss = float('inf')
            avg_metrics = {}

        # Validation
        avg_val_loss = None
        if epoch % eval_every == 0 and len(val_data['image']) > 0:
            print("Running validation...")
            wm.eval()
            val_losses = []
            val_batches = 0
            successful_val_batches = 0
            
            with torch.no_grad():
                for batch in create_data_loader(val_data, max(1, batch_size//2), max(16, batch_length//2), shuffle=False):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    batch['reward'] = torch.zeros(batch['action'].shape[0], batch['action'].shape[1], 
                                                device=device, dtype=torch.float32)
                    try:
                        # Simple validation loss computation
                        data_processed = wm.preprocess(batch)
                        embed = wm.encoder(data_processed)
                        post, prior = wm.dynamics.observe(
                            embed, data_processed["action"], data_processed["is_first"]
                        )
                        kl_loss, kl_value, dyn_loss, rep_loss = wm.dynamics.kl_loss(
                            post, prior,
                            wm._config.kl_free, wm._config.dyn_scale, wm._config.rep_scale
                        )
                        val_loss = torch.mean(kl_loss).item()
                        val_losses.append(val_loss)
                        successful_val_batches += 1
                    except Exception as e:
                        print(f"Validation error in batch {val_batches}: {e}")
                        torch.cuda.empty_cache()
                        
                    val_batches += 1
                    if val_batches >= 10:  # Limit validation batches
                        break

            if val_losses:
                avg_val_loss = np.mean(val_losses)
                writer.add_scalar('val/loss', avg_val_loss, epoch)
                print(f"Validation loss: {avg_val_loss:.4f} (from {successful_val_batches}/{val_batches} batches)")
                is_best = avg_val_loss < best_loss
                if is_best:
                    best_loss = avg_val_loss
            else:
                is_best = False
                print("No successful validation batches.")
        else:
            is_best = False

        # Save checkpoints
        if epoch % save_every == 0 or is_best or epoch == epochs - 1:
            metrics_to_save = avg_metrics.copy()
            if avg_val_loss is not None:
                metrics_to_save['val_loss'] = avg_val_loss
            try:
                checkpoint_path = save_checkpoint(wm, epoch, metrics_to_save, checkpoint_dir, is_best)
                print(f"Checkpoint saved: {checkpoint_path}")
            except Exception as e:
                print(f"Error saving checkpoint: {e}")

        # Final cleanup
        torch.cuda.empty_cache()

    writer.close()
    print(f"\nTraining complete! Best validation loss: {best_loss:.4f}")
    return wm
  
def load_multiple_hdf5_files(hdf5_paths, max_sequences_per_file=None, pad_sequences=True, max_length=None):

    
    all_data = {}
    total_sequences = 0
    max_seq_length = 0
    
    if pad_sequences and max_length is None:
        print("First pass: determining maximum sequence length...")
        for hdf5_path in hdf5_paths:
            try:
                data = convert_hdf5_to_dreamer(hdf5_path)
                if 'reward' in data:
                    del data['reward']
                seq_length = data['image'].shape[1]
                max_seq_length = max(max_seq_length, seq_length)
                print(f"  - {Path(hdf5_path).name}: {seq_length} timesteps")
            except Exception as e:
                print(f"Error checking {hdf5_path}: {e}")
        print(f"Maximum sequence length found: {max_seq_length}")
    elif max_length is not None:
        max_seq_length = max_length
        print(f"Using specified maximum length: {max_seq_length}")
    
    for i, hdf5_path in enumerate(hdf5_paths):
        print(f"Loading file {i+1}/{len(hdf5_paths)}: {hdf5_path}")
        
        try:
            data = convert_hdf5_to_dreamer(hdf5_path)
            
            # Remove reward if present
            if 'reward' in data:
                del data['reward']
            
            if max_sequences_per_file is not None:
                for key in data.keys():
                    data[key] = data[key][:max_sequences_per_file]
            
            current_length = data['image'].shape[1]
            
            if pad_sequences and current_length < max_seq_length:
                print(f"  - Padding sequences from {current_length} to {max_seq_length} timesteps")
                padded_data = {}
                for key, tensor in data.items():
                    if key in ['image', 'action', 'is_terminal', 'is_first', 'discount']:
                        pad_length = max_seq_length - current_length
                        
                        if key == 'image':
                            last_frame = tensor[:, -1:].expand(-1, pad_length, -1, -1, -1)
                            padded_data[key] = torch.cat([tensor, last_frame], dim=1)
                        elif key == 'action':
                            pad_shape = (tensor.shape[0], pad_length, tensor.shape[2])
                            pad_tensor = torch.zeros(pad_shape, dtype=tensor.dtype)
                            padded_data[key] = torch.cat([tensor, pad_tensor], dim=1)
                        elif key == 'is_terminal':
                            pad_shape = (tensor.shape[0], pad_length)
                            pad_tensor = torch.ones(pad_shape, dtype=tensor.dtype)
                            padded_data[key] = torch.cat([tensor, pad_tensor], dim=1)
                        elif key == 'is_first':
                            pad_shape = (tensor.shape[0], pad_length)
                            pad_tensor = torch.zeros(pad_shape, dtype=tensor.dtype)
                            padded_data[key] = torch.cat([tensor, pad_tensor], dim=1)
                        elif key == 'discount':
                            pad_shape = (tensor.shape[0], pad_length)
                            pad_tensor = torch.zeros(pad_shape, dtype=tensor.dtype)
                            padded_data[key] = torch.cat([tensor, pad_tensor], dim=1)
                        else:
                            padded_data[key] = tensor
                
                data = padded_data
            
            elif not pad_sequences and current_length != max_seq_length and i > 0:
                print(f"  - Truncating sequences from {current_length} to {max_seq_length} timesteps")
                for key in data.keys():
                    if data[key].ndim > 1 and data[key].shape[1] > max_seq_length:
                        data[key] = data[key][:, :max_seq_length]
            
            if i == 0:
                required_keys = ['image', 'action', 'is_terminal', 'is_first']
                if 'discount' in data:
                    required_keys.append('discount')
                
                all_data = {key: [data[key]] for key in required_keys}
                print(f"  - Loaded {data['image'].shape[0]} sequences")
                print(f"  - Sequence length: {data['image'].shape[1]}")
                print(f"  - Image shape: {data['image'].shape}")
                total_sequences += data['image'].shape[0]
            else:
                for key in all_data.keys():
                    if key not in data:
                        raise ValueError(f"Key '{key}' missing in file {hdf5_path}")
                    
                    if data[key].shape[1:] != all_data[key][0].shape[1:]:
                        raise ValueError(f"Shape mismatch for key '{key}' in file {hdf5_path}. "
                                       f"Expected {all_data[key][0].shape[1:]}, got {data[key].shape[1:]}")
                
                for key in all_data.keys():
                    all_data[key].append(data[key])
                
                print(f"  - Loaded {data['image'].shape[0]} sequences")
                print(f"  - Sequence length: {data['image'].shape[1]}")
                total_sequences += data['image'].shape[0]
                
        except Exception as e:
            print(f"Error loading {hdf5_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_data:
        raise ValueError("No valid HDF5 files could be loaded")
    
    print(f"Combining data from all files...")
    combined_data = {}
    for key in all_data.keys():
        combined_data[key] = torch.cat(all_data[key], dim=0)
        print(f"  - {key}: {combined_data[key].shape}")
    
    print(f"Total sequences loaded: {total_sequences}")
    print(f"Final sequence length: {combined_data['image'].shape[1]}")
    return combined_data


def load_multiple_hdf5_files_truncate(hdf5_paths, max_sequences_per_file=None, target_length=None):
    """Alternative approach: truncate all sequences to the shortest length"""
    print(f"Loading data from {len(hdf5_paths)} HDF5 files (truncate mode)...")
    
    all_data = {}
    total_sequences = 0
    min_seq_length = float('inf')
    
    # First pass: find minimum sequence length
    print("Finding minimum sequence length...")
    for hdf5_path in hdf5_paths:
        try:
            data = convert_hdf5_to_dreamer(hdf5_path)
            seq_length = data['image'].shape[1]
            min_seq_length = min(min_seq_length, seq_length)
            print(f"  - {Path(hdf5_path).name}: {seq_length} timesteps")
        except Exception as e:
            print(f"Error checking {hdf5_path}: {e}")
    
    if target_length is not None:
        min_seq_length = min(min_seq_length, target_length)
    
    print(f"Will truncate all sequences to: {min_seq_length} timesteps")
    
    # Second pass: load and truncate data
    for i, hdf5_path in enumerate(hdf5_paths):
        print(f"Loading file {i+1}/{len(hdf5_paths)}: {hdf5_path}")
        
        try:
            data = convert_hdf5_to_dreamer(hdf5_path)
            
            # Limit sequences per file if specified
            if max_sequences_per_file is not None:
                for key in data.keys():
                    data[key] = data[key][:max_sequences_per_file]
            
            # Truncate to minimum length
            for key in data.keys():
                if data[key].ndim > 1 and data[key].shape[1] > min_seq_length:
                    data[key] = data[key][:, :min_seq_length]
            
            # Initialize or append data
            if i == 0:
                all_data = {key: [value] for key, value in data.items()}
            else:
                for key in all_data.keys():
                    all_data[key].append(data[key])
            
            print(f"  - Loaded {data['image'].shape[0]} sequences")
            print(f"  - Truncated to length: {data['image'].shape[1]}")
            total_sequences += data['image'].shape[0]
                
        except Exception as e:
            print(f"Error loading {hdf5_path}: {e}")
            continue
    
    if not all_data:
        raise ValueError("No valid HDF5 files could be loaded")
    
    # Concatenate all data
    print(f"Combining data from all files...")
    combined_data = {}
    for key in all_data.keys():
        combined_data[key] = torch.cat(all_data[key], dim=0)
        print(f"  - {key}: {combined_data[key].shape}")
    
    print(f"Total sequences loaded: {total_sequences}")
    return combined_data

def get_hdf5_files_from_pattern(pattern_or_path):
    """Get list of HDF5 files from a pattern, directory, or single file"""
    if os.path.isfile(pattern_or_path):
        # Single file
        return [pattern_or_path]
    elif os.path.isdir(pattern_or_path):
        # Directory - find all HDF5 files
        patterns = ['*.hdf5', '*.h5']
        files = []
        for pattern in patterns:
            files.extend(glob.glob(os.path.join(pattern_or_path, pattern)))
        return sorted(files)
    else:
        # Glob pattern
        files = glob.glob(pattern_or_path)
        return sorted(files)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DreamerV3 WorldModel')
    parser.add_argument('--config', type=str, default='worldmodel_config.yaml',
                      help='Path to config YAML file')
    parser.add_argument('--hdf5', type=str, required=True,
                      help='Path to HDF5 file, directory, or glob pattern (e.g., "data/*.hdf5")')
    
    # Data loading parameters
    parser.add_argument('--max_sequences_per_file', type=int, default=None,
                      help='Maximum sequences per file (for memory management)')
    parser.add_argument('--shuffle_files', action='store_true', default=True,
                      help='Shuffle HDF5 files before loading')
    parser.add_argument('--no_shuffle_files', dest='shuffle_files', action='store_false',
                      help='Do not shuffle HDF5 files')
    parser.add_argument('--sequence_mode', type=str, choices=['pad', 'truncate'], default='pad',
                      help='How to handle different sequence lengths')
    parser.add_argument('--max_sequence_length', type=int, default=None,
                      help='Maximum sequence length to use')
    
    # Model and training parameters
    parser.add_argument('--size', type=str, default='1m',
                      help='Model size (e.g., 1m, 12m, 70m)')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                      help='Batch size for training')
    parser.add_argument('--batch_length', type=int, default=64,
                      help='Sequence length for training')
    
    # Logging and checkpoints
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                      help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs',
                      help='Directory to save logs')
    parser.add_argument('--resume_from', type=str, default=None,
                      help='Path to checkpoint to resume from')
    parser.add_argument('--save_every', type=int, default=10,
                      help='Save checkpoint every N epochs')
    parser.add_argument('--eval_every', type=int, default=5,
                      help='Run evaluation every N epochs')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                      help='Number of steps to accumulate gradients')

    args = parser.parse_args()
    
    train_worldmodel(
        config_path=args.config,
        hdf5_input=args.hdf5,
        size=args.size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        batch_length=args.batch_length,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        resume_from=args.resume_from,
        save_every=args.save_every,
        eval_every=args.eval_every,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_sequences_per_file=args.max_sequences_per_file,
        shuffle_files=args.shuffle_files,
        sequence_mode=args.sequence_mode,
        max_sequence_length=args.max_sequence_length
    )