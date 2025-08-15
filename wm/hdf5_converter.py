import h5py
import numpy as np
import torch

def convert_hdf5_to_dreamer(hdf5_path, camera_key='agentview_rgb'):
    """Convert HDF5 data to format expected by Dreamer WorldModel WITHOUT rewards and discount"""

    with h5py.File(hdf5_path, 'r') as f:
        # Collect all demos
        all_images = []
        all_actions = []
        all_dones = []

        for demo_key in f['data'].keys():
            base = f['data'][demo_key]

            images = base['obs'][camera_key][()]            # (T, H, W, C)
            actions = base['actions'][()]                   # (T, A)
            dones = base['dones'][()]                       # (T,)

            all_images.append(images)
            all_actions.append(actions)
            all_dones.append(dones)

        # Stack all episodes as batch dimension
        # Pad sequences to same length if needed
        max_length = max(len(ep) for ep in all_actions)

        batch_images = []
        batch_actions = []
        batch_dones = []
        batch_is_first = []

        for i, (images, actions, dones) in enumerate(zip(all_images, all_actions, all_dones)):
            T = len(actions)

            if T < max_length:
                pad_length = max_length - T
                images = np.concatenate([images, np.zeros((pad_length,) + images.shape[1:], dtype=images.dtype)])
                actions = np.concatenate([actions, np.zeros((pad_length,) + actions.shape[1:], dtype=actions.dtype)])
                dones = np.concatenate([dones, np.ones(pad_length, dtype=dones.dtype)])  

            is_first = np.zeros(max_length, dtype=bool)
            is_first[0] = True

            batch_images.append(images)
            batch_actions.append(actions)
            batch_dones.append(dones)
            batch_is_first.append(is_first)

        # Convert to tensors 
        obs = {
            'image': torch.from_numpy(np.stack(batch_images)).float(),      # (B, T, H, W, C)
            'action': torch.from_numpy(np.stack(batch_actions)).float(),    # (B, T, A)
            'is_terminal': torch.from_numpy(np.stack(batch_dones)).bool(),  # (B, T)
            'is_first': torch.from_numpy(np.stack(batch_is_first)).bool(),  # (B, T)
        }

        print(f"Loaded {len(all_images)} episodes")
        print(f"Data shapes (NO REWARDS/DISCOUNT):")
        for key, value in obs.items():
            print(f"  {key}: {value.shape}")

        # Print value ranges for debugging
        print(f"Image value range: [{obs['image'].min():.3f}, {obs['image'].max():.3f}]")
        print(f"Action value range: [{obs['action'].min():.3f}, {obs['action'].max():.3f}]")

        return obs