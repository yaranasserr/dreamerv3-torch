
import os
import torch
from datetime import datetime


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