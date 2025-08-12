Basic training on directory:
bashpython your_script.py --mode train \
    --hdf5 "/home/user/robot_data/" \
    --config worldmodel_config.yaml \
    --epochs 50 \
    --batch_size 16
Training with memory management:
bashpython your_script.py --mode train \
    --hdf5 "datasets/*.hdf5" \
    --config worldmodel_config.yaml \
    --max_sequences_per_file 1000 \
    --batch_size 8 \
    --epochs 100
Training with specific pattern:
bashpython your_script.py --mode train \
    --hdf5 "robot_logs/experiment_2024_*.hdf5" \
    --config worldmodel_config.yaml \
    --shuffle_files \
    --epochs 75
Training without file shuffling:
bashpython your_script.py --mode train \
    --hdf5 "/data/sequential_runs/" \
    --config worldmodel_config.yaml \
    --no_shuffle_files \
    --epochs 50