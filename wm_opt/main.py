#!/usr/bin/env python3
"""Main script for training DreamerV3 WorldModel"""

import argparse
from trainer import train_worldmodel


def main():
    parser = argparse.ArgumentParser(description='Train DreamerV3 WorldModel')
    parser.add_argument('--config', type=str, default='worldmodel_config.yaml',
                      help='Path to config YAML file')
    parser.add_argument('--hdf5', type=str, required=True,
                      help='Path to HDF5 file, directory, or glob pattern (e.g., "data/*.hdf5")')
    
    # Data loading parameters
    parser.add_argument('--max_sequences_per_file', type=int, default=None,
                      help='Maximum sequences per file (for memory management)')
    parser.add_argument('--shuffle_files', action='store_true', default=True,
                      help='Shuffle HDF5 files before loading')
    parser.add_argument('--no_shuffle_files', dest='shuffle_files', action='store_false',
                      help='Do not shuffle HDF5 files')
    parser.add_argument('--sequence_mode', type=str, choices=['pad', 'truncate'], default='pad',
                      help='How to handle different sequence lengths')
    parser.add_argument('--max_sequence_length', type=int, default=None,
                      help='Maximum sequence length to use')
    
    # Model and training parameters
    parser.add_argument('--size', type=str, default='1m',
                      help='Model size (e.g., 1m, 12m, 70m)')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                      help='Batch size for training')
    parser.add_argument('--batch_length', type=int, default=64,
                      help='Sequence length for training')
    
    # Logging and checkpoints
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                      help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs',
                      help='Directory to save logs')
    parser.add_argument('--resume_from', type=str, default=None,
                      help='Path to checkpoint to resume from')
    parser.add_argument('--save_every', type=int, default=10,
                      help='Save checkpoint every N epochs')
    parser.add_argument('--eval_every', type=int, default=5,
                      help='Run evaluation every N epochs')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                      help='Number of steps to accumulate gradients')

    args = parser.parse_args()
    
    # Train the model
    trained_model = train_worldmodel(
        config_path=args.config,
        hdf5_input=args.hdf5,
        size=args.size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        batch_length=args.batch_length,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        resume_from=args.resume_from,
        save_every=args.save_every,
        eval_every=args.eval_every,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_sequences_per_file=args.max_sequences_per_file,
        shuffle_files=args.shuffle_files,
        sequence_mode=args.sequence_mode,
        max_sequence_length=args.max_sequence_length
    )
    
    print("Training completed successfully!")


if __name__ == '__main__':
    main()