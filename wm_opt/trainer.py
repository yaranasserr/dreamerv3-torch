import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from gymnasium.spaces import Box, Dict
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import WorldModel


from wm_opt.data_loader import load_multiple_hdf5_files, load_multiple_hdf5_files_truncate, create_data_loader, get_hdf5_files_from_pattern
from wm_opt.config import Config, apply_size_override, setup_config_for_worldmodel,load_config
from wm_opt.checkpoint import load_checkpoint, save_checkpoint
from wm_opt.loss import compute_validation_loss_MSE_AWARE

def train_worldmodel(config_path, hdf5_input, size='1m', epochs=100, batch_size=16, 
                    batch_length=64, checkpoint_dir='./checkpoints', log_dir='./logs',
                    resume_from=None, save_every=10, eval_every=5, 
                    gradient_accumulation_steps=1, max_sequences_per_file=None,
                    shuffle_files=True, sequence_mode='pad', max_sequence_length=None):
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    # Setup logging
    writer = SummaryWriter(log_dir)
    # Load and setup config
    
    config = load_config(config_path, size=size, batch_size=batch_size, batch_length=batch_length)


    size_key = f"size{size}"
    apply_size_override(config, size_key)
    cfg_dict = config.to_dict()

    cfg_dict.update({'batch_size': batch_size, 'batch_length': batch_length})
    size_overrides = config.get(size_key, {})
    cfg_dict = setup_config_for_worldmodel(cfg_dict, size_overrides)

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

    
    if data['image'].max() <= 1.0:
        # If data is already normalized to [0,1], scale back to [0,255]
        data['image'] = data['image'] * 255.0
    
    data['image'] = data['image'].float()  
    data['action'] = data['action'].float()
    data['is_terminal'] = data['is_terminal'].bool()
    data['is_first'] = data['is_first'].bool()

    actual_action_dim = data['action'].shape[-1]
    cfg_dict['num_actions'] = actual_action_dim
    cfg = Config(**cfg_dict)
    device = cfg_dict.get('device', 'cuda')

    image_shape = data['image'].shape[2:]  # (H, W, C)
    obs_space = Dict({'image': Box(low=0, high=255, shape=image_shape, dtype=np.float32)})
    act_space = Box(low=-1.0, high=1.0, shape=(actual_action_dim,), dtype=np.float32)

    print("Initializing WorldModel...")
    try:
        wm = WorldModel(obs_space, act_space, 0, cfg).to(device)
        print("WorldModel initialized successfully!")
        print(f"Model parameters: {sum(p.numel() for p in wm.parameters()):,}")
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
        best_loss = last_metrics.get('val_loss', float('inf'))
        start_epoch += 1

    # Split data into train/validation
    val_split = 0.1
    num_sequences = data['image'].shape[0]
    val_size = int(num_sequences * val_split)
    indices = torch.randperm(num_sequences)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    # Create train/val splits
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

            
            try:
                # Use the original _train method
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

        avg_val_loss = None
        if epoch % eval_every == 0 and len(val_data['image']) > 0:
            print("Running validation...")
            wm.eval()
            val_metrics_list = []
            val_batches = 0
            successful_val_batches = 0
            
            for i, batch in enumerate(create_data_loader(val_data, max(1, batch_size//2), max(16, batch_length//2), shuffle=False)):
                batch = {k: v.to(device) for k, v in batch.items()}
                batch['reward'] = torch.zeros(batch['action'].shape[0], batch['action'].shape[1], 
                                            device=device, dtype=torch.float32)
            
                val_metrics = compute_validation_loss_MSE_AWARE(wm, batch)
                if val_metrics is not None:
                    val_metrics_list.append(val_metrics)
                    successful_val_batches += 1
                        
                val_batches += 1
                if val_batches >= 10:  
                    break

            if val_metrics_list:
                
                avg_val_metrics = {}
                for key in val_metrics_list[0].keys():
                    avg_val_metrics[key] = np.mean([m[key] for m in val_metrics_list])
                
                avg_val_loss = avg_val_metrics['total_loss']
                
                for k, v in avg_val_metrics.items():
                    writer.add_scalar(f'val/{k}', v, epoch)
                
                print(f"Validation loss: {avg_val_loss:.4f} (from {successful_val_batches}/{val_batches} batches)")
                print(f"  - KL loss: {avg_val_metrics['kl_loss']:.4f}")
                print(f"  - Reconstruction loss: {avg_val_metrics.get('image_loss', 0):.4f}")
                
                is_best = avg_val_loss < best_loss
                if is_best:
                    best_loss = avg_val_loss
          
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
