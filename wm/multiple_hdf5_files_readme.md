# DreamerV3 WorldModel Training & Visualization

A comprehensive training and visualization script for DreamerV3 WorldModel with support for both single and multiple HDF5 files, automatic GPU memory optimization, and checkpoint management.
##  Initial Setup

Before using this training and visualization script, make sure to:

1. **Clone the DreamerV3 Torch repository**  
   This project builds on top of the PyTorch implementation of DreamerV3 by [NM512](https://github.com/NM512/dreamerv3-torch).

   ```bash
   git clone https://github.com/NM512/dreamerv3-torch.git
   cd dreamerv3-torch

Install dependencies using Python 3.11



```
pip install -r requirements.txt
```

Add the following custom files to the cloned repo root directory:

multiple_hdf5.py

memor_opt_training.py

training_wm_v2.py

hdf5_converter.py
## Features

- **Single & Multi-File Training**: Train on individual HDF5 files or entire directories
- **Pattern Matching**: Use wildcards and patterns to select specific datasets
- **Memory Optimization**: Automatic GPU memory management for limited hardware
- **Checkpoint System**: Save/load models with best model tracking
- **Visualization Tools**: Generate comparison videos and image grids
- **Multi-Size Support**: 1M, 12M, 70M parameter configurations
- **TensorBoard Logging**: Real-time training monitoring
- **File Management**: Shuffle files, limit sequences per file, directory scanning

##  Requirements

```bash
pip install torch torchvision
pip install gymnasium
pip install numpy matplotlib opencv-python
pip install tensorboard
pip install h5py yaml
```



### Multiple Files Training
```bash
python training_wm_v2.py --mode train \
  --hdf5 "/home/user/robot_data/" \
  --size 1m \
  --epochs 50
```

## 📖 Command Reference

### Core Arguments

| Argument | Description | Default | Example |
|----------|-------------|---------|---------|
| `--mode` | Operation mode: `train` or `visualize` | **Required** | `--mode train` |
| `--hdf5` | Path to HDF5 file, directory, or pattern | **Required** | `--hdf5 data.hdf5` |
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

### Multi-File Arguments

| Argument | Description | Default | Example |
|----------|-------------|---------|---------|
| `--max_sequences_per_file` | Limit sequences loaded per file | `None` | `--max_sequences_per_file 1000` |
| `--shuffle_files` | Shuffle file order during training | `True` | `--shuffle_files` |
| `--no_shuffle_files` | Disable file shuffling | `False` | `--no_shuffle_files` |



## 🎛️ Model Size Configurations

| Size | Parameters | GPU Memory |
|------|-----------|------------|
| `1m` | ~1.9M | 2-4GB | 
| `12m` | ~12M | 6-8GB | 
| `70m` | ~70M | 12-16GB |


### Memory-Optimized Training (4GB GPU)
```bash
# Optimized for limited GPU memory
python training_wm_v2.py --mode train \
  --hdf5 robot_data.hdf5 \
  --size 1m \
  --epochs 100 \
  --batch_size 4 \
  --batch_length 32 \
  --save_every 5 \
  --eval_every 10
```


## Multiple Files Training Examples

### Basic Directory Training
```bash

python training_wm_v2.py --mode train \
  --hdf5 "/home/user/robot_data/" \
  --config worldmodel_config.yaml \
  --epochs 50 \
  --batch_size 16
```

### Pattern-Based Training
```bash
python training_wm_v2.py --mode train \
  --hdf5 "datasets/*.hdf5" \
  --config worldmodel_config.yaml \
  --epochs 75 \
  --batch_size 8
```



### Sequential Training (No Shuffling)
```bash
# Train without file shuffling for temporal consistency
python training_wm_v2.py --mode train \
  --hdf5 "/data/sequential_runs/" \
  --config worldmodel_config.yaml \
  --no_shuffle_files \
  --epochs 50
```



## Visualization Examples

### Single File Visualization
```bash
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


## Resume Training Examples


### Resume Multi-File Training
```bash
# Resume training on directory
python training_wm_v2.py --mode train \
  --hdf5 "/home/user/robot_data/" \
  --resume_from ./checkpoints/best_model.pt \
  --epochs 200 \
  --batch_size 8
```


### Wildcard Patterns
```bash
--hdf5 "datasets/*.hdf5"
--hdf5 "robot_logs/exp_*.hdf5"
--hdf5 "/data/2024_*/*.hdf5"
```

### Specific Patterns
```bash
--hdf5 "experiment_2024_01_*.hdf5"
--hdf5 "robot_data/session_[0-9]*.hdf5"
--hdf5 "*/validation_*.hdf5"
```






### TensorBoard
```bash
# Start TensorBoard server
tensorboard --logdir ./logs

# Open in browser: http://localhost:6006
```
