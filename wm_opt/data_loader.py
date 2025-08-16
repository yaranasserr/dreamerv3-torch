"""Data loading utilities for DreamerV3 WorldModel training"""

import os
import glob
import numpy as np
import torch
from pathlib import Path
from hdf5_converter_v2 import convert_hdf5_to_dreamer


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


def load_multiple_hdf5_files(hdf5_paths, max_sequences_per_file=None, pad_sequences=True, max_length=None):
    """Load and combine multiple HDF5 files with sequence padding"""
    
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


def prepare_data(hdf5_input, max_sequences_per_file=None, shuffle_files=True, 
                sequence_mode='pad', max_sequence_length=None):
    """Main function to prepare data for training"""
    
    hdf5_files = get_hdf5_files_from_pattern(hdf5_input)
    
    if shuffle_files:
        import random
        random.shuffle(hdf5_files)

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
    
    # Normalize and prepare data
    if data['image'].max() <= 1.0:
        # If data is already normalized to [0,1], scale back to [0,255]
        data['image'] = data['image'] * 255.0
    
    data['image'] = data['image'].float()  
    data['action'] = data['action'].float()
    data['is_terminal'] = data['is_terminal'].bool()
    data['is_first'] = data['is_first'].bool()
    
    return data