

import torch
import numpy as np


def compute_validation_loss(wm, batch):
   
    try:
        with torch.no_grad():
            data = wm.preprocess(batch)
            embed = wm.encoder(data)
            
            post, prior = wm.dynamics.observe(
                embed, data["action"], data["is_first"]
            )
            
            # Compute KL loss
            kl_free = wm._config.kl_free
            dyn_scale = wm._config.dyn_scale
            rep_scale = wm._config.rep_scale
            kl_loss, kl_value, dyn_loss, rep_loss = wm.dynamics.kl_loss(
                post, prior, kl_free, dyn_scale, rep_scale
            )
            
            # Compute reconstruction losses for all heads
            preds = {}
            losses = {}
            for name, head in wm.heads.items():
                # Always use gradients for validation to get proper predictions
                feat = wm.dynamics.get_feat(post)
                pred = head(feat)
                if type(pred) is dict:
                    preds.update(pred)
                else:
                    preds[name] = pred
            
            # Compute prediction losses (negative log likelihood)
            for name, pred in preds.items():
                if name in data:  # Only compute loss for available data
                    loss = -pred.log_prob(data[name])
                    losses[name] = loss
            
            # Apply loss scales (same as training)
            scaled_losses = {
                key: value * wm._scales.get(key, 1.0)
                for key, value in losses.items()
            }
            
            # Total model loss (same formula as training)
            model_loss = sum(scaled_losses.values()) + kl_loss
            total_loss = torch.mean(model_loss)
            
            # Return individual loss components for logging
            metrics = {
                'total_loss': total_loss.item(),
                'kl_loss': torch.mean(kl_loss).item(),
                'dyn_loss': torch.mean(dyn_loss).item(),
                'rep_loss': torch.mean(rep_loss).item(),
            }
            
            # Add individual head losses
            for name, loss in losses.items():
                metrics[f'{name}_loss'] = torch.mean(loss).item()
            
            return metrics
            
    except Exception as e:
        print(f"Validation error: {e}")
        return None


def compute_validation_loss_MSE_AWARE(wm, batch):

    try:
        with torch.no_grad():
            data = wm.preprocess(batch)
            embed = wm.encoder(data)
            post, prior = wm.dynamics.observe(embed, data["action"], data["is_first"])
            
            # KL loss (same as before)
            kl_free = wm._config.kl_free
            dyn_scale = wm._config.dyn_scale
            rep_scale = wm._config.rep_scale
            kl_loss, kl_value, dyn_loss, rep_loss = wm.dynamics.kl_loss(
                post, prior, kl_free, dyn_scale, rep_scale
            )
            
            # Head predictions (same as training)
            preds = {}
            for name, head in wm.heads.items():
                grad_head = name in wm._config.grad_heads
                feat = wm.dynamics.get_feat(post)
                feat = feat if grad_head else feat.detach()
                pred = head(feat)
                if type(pred) is dict:
                    preds.update(pred)
                else:
                    preds[name] = pred
            
            # Compute losses (same as training)
            losses = {}
            for name, pred in preds.items():
                if name in data:
                    loss = -pred.log_prob(data[name])
                    losses[name] = loss
            
            # Apply scales (same as training) 
            scaled = {
                key: value * wm._scales.get(key, 1.0)
                for key, value in losses.items()
            }
            
            model_loss = sum(scaled.values()) + kl_loss
            total_loss = torch.mean(model_loss)
            
            metrics = {
                'total_loss': total_loss.item(),
                'kl_loss': torch.mean(kl_loss).item(),
                'dyn_loss': torch.mean(dyn_loss).item(),
                'rep_loss': torch.mean(rep_loss).item(),
            }
            
            for name, loss in losses.items():
                metrics[f'{name}_loss'] = torch.mean(loss).item()
                
            # Debug print for first few calls
            if not hasattr(compute_validation_loss_MSE_AWARE, 'call_count'):
                compute_validation_loss_MSE_AWARE.call_count = 0
            compute_validation_loss_MSE_AWARE.call_count += 1
            
            if compute_validation_loss_MSE_AWARE.call_count <= 3:
                print(f"Validation debug call {compute_validation_loss_MSE_AWARE.call_count}:")
                print(f"  Total loss: {total_loss.item():.3f}")
                for name, loss in losses.items():
                    if wm._scales.get(name, 1.0) > 0:
                        print(f"  {name} loss: {torch.mean(loss).item():.3f} (scale: {wm._scales.get(name, 1.0)})")
            
            return metrics
            
    except Exception as e:
        print(f"Validation error: {e}")
        return None