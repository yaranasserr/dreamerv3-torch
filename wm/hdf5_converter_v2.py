import h5py
import numpy as np
import torch

def convert_hdf5_to_dreamer(hdf5_path, image_key='agentview_image', done_key='terminated'):
    """
    Convert HDF5 dataset to Dreamer format:
    - image: (B, T, H, W, C)
    - action: (B, T, A)
    - is_first: (B, T)
    - is_terminal: (B, T)
    """
    with h5py.File(hdf5_path, 'r') as f:
        all_images = []
        all_actions = []
        all_terminated = []

        for demo_key in f['data'].keys():
            demo = f['data'][demo_key]

            images = demo['obs'][image_key][()]      # (T, H, W, C)
            actions = demo['actions'][()]            # (T, A)
            terminated = demo[done_key][()]          # (T,)

            all_images.append(images)
            all_actions.append(actions)
            all_terminated.append(terminated)

        # Pad sequences to max length
        max_length = max(len(ep) for ep in all_actions)

        batch_images = []
        batch_actions = []
        batch_terminated = []
        batch_is_first = []

        for images, actions, terminated in zip(all_images, all_actions, all_terminated):
            T = len(actions)
            if T < max_length:
                pad_len = max_length - T
                images = np.concatenate([images, np.zeros((pad_len,) + images.shape[1:], dtype=images.dtype)])
                actions = np.concatenate([actions, np.zeros((pad_len,) + actions.shape[1:], dtype=actions.dtype)])
                terminated = np.concatenate([terminated, np.ones(pad_len, dtype=terminated.dtype)])

            is_first = np.zeros(max_length, dtype=bool)
            is_first[0] = True

            batch_images.append(images)
            batch_actions.append(actions)
            batch_terminated.append(terminated)
            batch_is_first.append(is_first)

        # Convert to tensors
        obs = {
            'image': torch.from_numpy(np.stack(batch_images)).float(),       # (B, T, H, W, C)
            'action': torch.from_numpy(np.stack(batch_actions)).float(),     # (B, T, A)
            'is_terminal': torch.from_numpy(np.stack(batch_terminated)).bool(), # (B, T)
            'is_first': torch.from_numpy(np.stack(batch_is_first)).bool(),   # (B, T)
        }

        print(f"Loaded {len(all_images)} episodes")
        print("Data shapes:")
        for key, value in obs.items():
            print(f"  {key}: {value.shape}")

        return obs
