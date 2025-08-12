# DreamerV3 WorldModel Training & Visualization

A comprehensive training and visualization script for DreamerV3 WorldModel with automatic GPU memory optimization and checkpoint management.

## 🚀 Features

- **Full Training Pipeline**: Train WorldModel from scratch with configurable parameters
- **Memory Optimization**: Automatic GPU memory management for limited hardware
- **Checkpoint System**: Save/load models with best model tracking
- **Visualization Tools**: Generate comparison videos and image grids
- **Multi-Size Support**: 1M, 12M, 70M parameter configurations
- **TensorBoard Logging**: Real-time training monitoring
- **Error Recovery**: Robust handling of GPU memory issues

## 📋 Requirements

```bash
pip install torch torchvision
pip install gymnasium
pip install numpy matplotlib opencv-python
pip install tensorboard
pip install h5py yaml
```

## 🎯 Quick Start

### 1. Training a New Model
```bash
python wm_training.py --mode train \
  --hdf5 /path/to/your/data.hdf5 \
  --size 1m \
  --epochs 50
```

### 2. Visualizing Results
```bash
python wm_training.py --mode visualize \
  --hdf5 /path/to/your/data.hdf5 \
  --checkpoint ./checkpoints/best_model.pt \
  --num_sequences 5
```

## 📖 Command Reference

### Core Arguments

| Argument | Description | Default | Example |
|----------|-------------|---------|---------|
| `--mode` | Operation mode: `train` or `visualize` | **Required** | `--mode train` |
| `--hdf5` | Path to HDF5 data file | **Required** | `--hdf5 data.hdf5` |
| `--config` | Path to config YAML file | `worldmodel_config.yaml` | `--config my_config.yaml` |
| `--size` | Model size configuration | `1m` | `--size 12m` |

### Training Arguments

| Argument | Description | Default | Recommended for 4GB GPU |
|----------|-------------|---------|-------------------------|
| `--epochs` | Number of training epochs | `100` | `50-100` |
| `--batch_size` | Batch size for training | `16` | `4-8` |
| `--batch_length` | Sequence length for training | `64` | `32` |
| `--checkpoint_dir` | Directory to save checkpoints | `./checkpoints` | `./checkpoints` |
| `--log_dir` | Directory to save TensorBoard logs | `./logs` | `./logs` |
| `--resume_from` | Path to checkpoint to resume from | `None` | `./checkpoints/best_model.pt` |
| `--save_every` | Save checkpoint every N epochs | `10` | `5-10` |
| `--eval_every` | Run validation every N epochs | `5` | `5-10` |
| `--gradient_accumulation_steps` | Steps to accumulate gradients | `1` | `1-2` |

### Visualization Arguments

| Argument | Description | Default | Example |
|----------|-------------|---------|---------|
| `--checkpoint` | Path to trained model checkpoint | `None` | `./checkpoints/best_model.pt` |
| `--output_dir` | Output directory for visualizations | `./visualizations` | `./outputs` |
| `--sequence_idx` | Starting sequence index | `0` | `--sequence_idx 10` |
| `--num_sequences` | Number of sequences to visualize | `1` | `--num_sequences 5` |
| `--no_videos` | Skip saving videos (only grids) | `False` | `--no_videos` |
| `--no_grids` | Skip saving grids (only videos) | `False` | `--no_grids` |

## 🎛️ Model Size Configurations

| Size | Parameters | GPU Memory | Use Case |
|------|-----------|------------|----------|
| `1m` | ~1.9M | 2-4GB | Learning, testing, limited hardware |
| `12m` | ~12M | 6-8GB | Good quality, balanced performance |
| `70m` | ~70M | 12-16GB | Best quality, research |

## 💾 Usage Examples

### Basic Training
```bash
# Train 1M model for 50 epochs
python wm_training.py --mode train \
  --hdf5 robot_data.hdf5 \
  --size 1m \
  --epochs 50 \
  --batch_size 8 \
  --batch_length 32
```

### Memory-Optimized Training (4GB GPU)
```bash
# Optimized for limited GPU memory
python wm_training.py --mode train \
  --hdf5 robot_data.hdf5 \
  --size 1m \
  --epochs 100 \
  --batch_size 4 \
  --batch_length 32 \
  --save_every 5 \
  --eval_every 10
```

### Resume Training
```bash
# Continue training from checkpoint
python wm_training.py --mode train \
  --hdf5 robot_data.hdf5 \
  --resume_from ./checkpoints/checkpoint_epoch_20.pt \
  --epochs 100
```

### High-Quality Training (12M model)
```bash
# Train larger model for better quality
python wm_training.py --mode train \
  --hdf5 robot_data.hdf5 \
  --size 12m \
  --epochs 100 \
  --batch_size 16 \
  --batch_length 64
```

### Comprehensive Visualization
```bash
# Generate visualizations for multiple sequences
python wm_training.py --mode visualize \
  --hdf5 robot_data.hdf5 \
  --checkpoint ./checkpoints/best_model.pt \
  --size 1m \
  --num_sequences 10 \
  --sequence_idx 0 \
  --output_dir ./results
```

### Visualization-Only (No Videos)
```bash
# Generate only comparison grids
python wm_training.py --mode visualize \
  --hdf5 robot_data.hdf5 \
  --checkpoint ./checkpoints/best_model.pt \
  --no_videos \
  --num_sequences 5
```

## 📊 Output Files

### Training Outputs
```
./checkpoints/
├── best_model.pt              # Best performing model
├── checkpoint_epoch_0.pt      # Regular checkpoints
├── checkpoint_epoch_10.pt
└── ...

./logs/
├── events.out.tfevents.*      # TensorBoard logs
└── ...
```

### Visualization Outputs
```
./visualizations/
├── comparison_grid_seq_0.png  # Original vs reconstructed frames
├── comparison_grid_seq_1.png
├── original_seq_0.mp4         # Original video sequences
├── reconstructed_seq_0.mp4    # Model reconstructions
├── comparison_seq_0.mp4       # Side-by-side comparisons
└── ...
```

## 🔧 Monitoring Training

### TensorBoard
```bash
# Start TensorBoard server
tensorboard --logdir ./logs

# Open in browser: http://localhost:6006
```

### Training Metrics
- **model_loss**: Overall reconstruction + dynamics loss
- **image_loss**: Image reconstruction quality
- **reward_loss**: Reward prediction accuracy
- **kl**: KL divergence (regularization)
- **val_loss**: Validation performance

### Loss Interpretation
```
Good Training Signs:
✅ Decreasing loss over time
✅ Validation loss following training loss
✅ No sudden spikes or divergence

Typical Loss Ranges:
- Initial: 1000-10000
- Converged 1M: 0.5-1.0
- Converged 12M: 0.3-0.7
- Converged 70M: 0.2-0.5
```

## 🚨 Troubleshooting

### GPU Out of Memory
```bash
# Reduce batch size and length
python wm_training.py --mode train \
  --batch_size 4 \
  --batch_length 16 \
  --hdf5 data.hdf5
```

### Training Slow/Unstable
```bash
# Use gradient accumulation
python wm_training.py --mode train \
  --batch_size 4 \
  --gradient_accumulation_steps 2 \
  --hdf5 data.hdf5
```

### Poor Reconstruction Quality
```bash
# Try larger model
python wm_training.py --mode train \
  --size 12m \
  --epochs 100 \
  --hdf5 data.hdf5

# Or train longer
python wm_training.py --mode train \
  --epochs 200 \
  --hdf5 data.hdf5
```

### Checkpoint Loading Issues
```bash
# Verify checkpoint path
ls -la ./checkpoints/

# Use absolute path
python wm_training.py --mode visualize \
  --checkpoint /full/path/to/best_model.pt \
  --hdf5 data.hdf5
```

## 🎯 Recommended Workflows

### For Beginners
1. Start with 1M model: `--size 1m --epochs 50`
2. Use small batches: `--batch_size 4 --batch_length 32`
3. Monitor with TensorBoard
4. Visualize results after training

### For Better Quality
1. Train 1M model first to verify setup
2. Scale up to 12M: `--size 12m --epochs 100`
3. Use larger batches if GPU allows
4. Train for more epochs

### For Production
1. Use 70M model for best quality
2. Train for 200+ epochs
3. Use validation to prevent overfitting
4. Save multiple checkpoints

## 🔬 Understanding Your Data

### Data Requirements
- **Format**: HDF5 with specific structure
- **Keys**: `image`, `action`, `reward`, `is_first`, `is_terminal`, `discount`
- **Image**: Shape (episodes, timesteps, height, width, channels)
- **Action**: Shape (episodes, timesteps, action_dim)

### Data Shapes Example
```
image: torch.Size([50, 259, 128, 128, 3])     # 50 episodes, 259 steps, 128x128 RGB
action: torch.Size([50, 259, 7])              # 7-dimensional action space
reward: torch.Size([50, 259])                 # Scalar rewards
```

## 📈 Performance Tips

### Memory Optimization
- Script automatically reduces batch size for <6GB GPUs
- Use `--batch_size 4 --batch_length 32` for 4GB GPUs
- Clear cache periodically (handled automatically)

### Training Speed
- Use `--gradient_accumulation_steps` for small batches
- Reduce `--eval_every` for faster training
- Use mixed precision (enabled by default)

