# Example of how to extend your test into a training loop:

def train_worldmodel(config_path, hdf5_path, size='1m', num_steps=1000):
    """Train the WorldModel for multiple steps"""
    
    # ... your existing setup code ...
    
    print(f"Training for {num_steps} steps...")
    
    for step in range(num_steps):
        # Sample random batch from data
        batch_indices = torch.randint(0, data['image'].shape[0], (batch_size,))
        time_indices = torch.randint(0, data['image'].shape[1] - batch_length, (batch_size,))
        
        batch_data = {}
        for key in ['image', 'action', 'reward', 'is_first', 'is_terminal', 'discount']:
            batch_data[key] = torch.stack([
                data[key][batch_indices[i], time_indices[i]:time_indices[i]+batch_length]
                for i in range(batch_size)
            ]).to(device)
        
        # Training step
        try:
            post, context, metrics = wm._train(batch_data)
            
            # Log progress
            if step % 100 == 0:
                print(f"Step {step}:")
                print(f"  Image Loss: {metrics['image_loss']:.2f}")
                print(f"  Reward Loss: {metrics['reward_loss']:.4f}")
                print(f"  KL: {metrics['kl']:.4f}")
                
                # Optional: Generate video prediction every 100 steps
                if step % 500 == 0:
                    video_pred = wm.video_pred(batch_data)
                    print(f"  Video prediction shape: {video_pred.shape}")
                    
        except Exception as e:
            print(f"Error at step {step}: {e}")
            break
    
    print("Training completed!")
    return wm
