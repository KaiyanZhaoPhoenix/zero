"""
Script to train a physics-informed neural network model (PhysNet or PhysRegMLP) 
for predicting building temperature dynamics.

This script performs the following steps:
1. Parses command-line arguments to configure the training process.
2. Loads and preprocesses the training data.
3. Initializes the model based on the selected model type.
4. Trains the model using the PyTorch Lightning Trainer.
5. Saves the trained model weights.

How to run:
python your_script_name.py --model_type PhysNet --depth 8 --epochs 75 --lr 0.001
"""

import argparse
import logging
import warnings
import os

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl

# Import utility functions and classes from your project
from utils.support_functions import transform_temp, transform_action, transform_outside_temp
from utils.global_state_variables import MAX_TIME
from utils.PhysNet import PhysNet
from utils.PhysRegMLP import PhysRegMLP
from utils.PhysMLP import PhysMLP

# Suppress warnings from PyTorch Lightning about GPU availability, etc.
warnings.filterwarnings('ignore')
logging.getLogger('lightning').setLevel(logging.ERROR)

# -------------------------------------------------------------------------------------------------------------------- #

def prepare_data(data: dict, depth: int) -> dict:
    """
    Preprocesses and scales the input and label data.

    Args:
        data (dict): A dictionary containing 'x_agg_k' and 'label_k'.
        depth (int): The depth of historical states to use.

    Returns:
        dict: A dictionary containing processed training samples (k) and next-step samples (k+1).
    """
    x_inputs = data['x_agg_k'].copy()
    y_labels = data['label_k'].copy()

    if x_inputs.shape[0] != y_labels.shape[0]:
        raise ValueError("The number of samples for inputs and labels must be the same.")

    state_depth = depth

    # Scale and transform features
    x_inputs[:, 0] = x_inputs[:, 0] / MAX_TIME  # Time
    x_inputs[:, 1] = transform_temp(x_inputs[:, 1])  # Current Temp
    x_inputs[:, 2:(2 + state_depth)] = transform_temp(x_inputs[:, 2:(2 + state_depth)])  # Previous temp states
    x_inputs[:, (2 + state_depth):(2 + 2 * state_depth)] = transform_action(
        x_inputs[:, (2 + state_depth):(2 + 2 * state_depth)])  # Previous Actions
    x_inputs[:, -2] = transform_action(x_inputs[:, -2])  # Current Action
    x_inputs[:, -1] = transform_outside_temp(x_inputs[:, -1])  # Outside Temp

    y_labels[:, 0] = transform_temp(y_labels[:, 0])
    y_labels[:, 1] = transform_action(y_labels[:, 1])

    return {
        "x_agg_k": x_inputs[:-1],
        "label_k": y_labels[:-1],
        "x_agg_k1": x_inputs[1:],
        "label_k1": y_labels[1:]
    }


def prepare_test_data(test_data: np.ndarray) -> dict:
    """
    (This function is not used in the main training script)
    Prepares test data for plotting or evaluation.
    """
    temp_track = []
    time_track = []
    outside_temperature = []
    u_phys_track = []
    u_track = []
    plotting_time_track = []
    day_i = 0
    quarter_i = 0

    for index in range(test_data.shape[0]):
        temp_track.append(test_data[index, 1])
        time_track.append(test_data[index, 0])
        u_phys_track.append(test_data[index, -2])
        u_track.append(test_data[index, 2])
        outside_temperature.append(test_data[index, -1])
        plotting_time_track.append(day_i * 24 + (quarter_i * 30) / 60)

        quarter_i += 1
        if quarter_i % 48 == 0:
            day_i += 1
            quarter_i = 0

    return {
        'temp_track': temp_track,
        'building_mass_temp_track': [], # Empty in the original script
        'time_track': time_track,
        'plotting_time_track': plotting_time_track,
        'u_phys_track': u_phys_track,
        'u_track': u_track,
        'outside_temperature': outside_temperature
    }


def get_model_config(args: argparse.Namespace) -> tuple:
    """
    Returns the model class and its corresponding network configuration based on command-line arguments.

    Args:
        args (argparse.Namespace): An object containing all command-line arguments.

    Returns:
        tuple: (Model class, Dictionary of network parameters)
    """
    if args.model_type == 'PhysRegMLP':
        model_class = PhysRegMLP
        network_param = {
            'lr': args.lr,
            'batch_size': args.batch_size,
            'lambda_value': args.lambda_2,
            'mdp_network': {
                'input_size': 4 + 2 * args.depth,
                'fc': [64] * 2,
                'output_size': 3,
                'activation': 'tanh',
                'dropout_rate': 0.01
            }
        }
    elif args.model_type == 'PhysMLP':
        model_class = PhysMLP
        network_param = {
            'lr': args.lr,
            'batch_size': args.batch_size,
            'lambda_value': args.lambda_2,
            'mdp_network': {
                'input_size': 4 + 2 * args.depth,
                'fc': [64] * 4,
                'output_size': 3,
                'activation': ['relu', 'tanh', 'relu', 'tanh'],
                'dropout_rate': 0.01
            }
        }
    elif args.model_type == 'PhysNet':
        model_class = PhysNet
        network_param = {
            'lr': args.lr,
            'batch_size': args.batch_size,
            'lambda_value': args.lambda_2,
            'encoding_network': {
                'input_size': 2 * args.depth,
                'fc': [24] * 1,
                'output_size': 1,
                'activation': 'tanh',
                'dropout_rate': 0.01
            },
            'mdp_network': {
                'input_size': 5,
                'fc': [128] * 1,
                'output_size': 2,
                'activation': 'tanh',
                'dropout_rate': 0.05
            }
        }
    else:
        raise TypeError(f"Unsupported model type: {args.model_type}")

    return model_class, network_param


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Train a physics-informed neural network model.")

    # Model-related arguments
    parser.add_argument('--model_type', type=str, default='PhysRegMLP', choices=['PhysNet','PhysMLP','PhysRegMLP'],
                        help='Type of model to train (default: PhysNet)')
    parser.add_argument('--depth', type=int, default=8,
                        help='Depth of historical data to use for state construction (default: 8)')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed for reproducibility (default: 1)')

    # Training hyperparameters
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Batch size for training (default: 2048)')
    parser.add_argument('--epochs', type=int, default=75,
                        help='Total number of training epochs (default: 75)')
    parser.add_argument('--lambda_2', type=float, default=0.5,
                        help='Regularization/weighting coefficient in the loss function (default: 0.5)')

    # Data and path arguments
    parser.add_argument('--data_path', type=str, default='data/Training_data.csv',
                        help='Path to the training data CSV file (default: data/Training_data.csv)')
    parser.add_argument('--output_path_template', type=str, default='models/{model_type}_depth{depth}_epochs{epochs}_seed{seed}.pth',
                        help='Template for the output path to save the trained model')
    
    # Compute device arguments
    parser.add_argument('--accelerator', type=str, default='cpu', choices=['cpu', 'gpu', 'auto'],
                        help="Select compute device ('cpu', 'gpu', 'auto') (default: auto)")
    parser.add_argument('--devices', type=int, default=1,
                        help="Number of devices to use (default: 1)")

    return parser.parse_args()


def main(args: argparse.Namespace):
    """
    The main training pipeline.
    """
    # 1. Set random seed for reproducibility
    pl.seed_everything(args.seed, workers=True)

    # 2. Load and prepare data
    print("Loading and preparing data...")
    training_data_df = pd.read_csv(args.data_path)
    training_data_main = training_data_df.to_numpy()

    # Select required feature columns: [time, x_k, prev_states, prev_actions, u_k, Ta_k]
    training_data_index_selection = tuple(
        [0, 1, *list(np.arange(4, 4 + args.depth)), *list(np.arange(4 + 24, 4 + 24 + args.depth)), 2, -1]
    )

    input_data_dict = {
        'x_agg_k': training_data_main[:, training_data_index_selection],
        'label_k': training_data_main[:, (3, -2)]  # [x_k+1, u_phys_k]
    }

    training_data_dict = prepare_data(input_data_dict, args.depth)
    print("Data preparation complete.")

    # 3. Initialize the model
    print(f"Initializing model: {args.model_type}...")
    model_class, network_param = get_model_config(args)
    model_instance = model_class(network_param)
    model_instance.add_training_data(training_data_dict)

    # 4. Initialize the PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        min_epochs=1,
        accelerator=args.accelerator,
        devices=args.devices,
        enable_checkpointing=False,  # Disable automatic checkpointing
        logger=True,  # Disable loggers
        enable_progress_bar=True, # Explicitly enable the progress bar
    )

    # 5. Train the model
    print("Starting model training...")
    trainer.fit(model_instance)
    print("Model training finished.")

    # 6. Save the model
    model_path = args.output_path_template.format(
        model_type=args.model_type,
        depth=args.depth,
        seed=args.seed,
        epochs= args.epochs,
    )

    output_dir = os.path.dirname(model_path)
    if output_dir:  # Check if the path includes a directory
        os.makedirs(output_dir, exist_ok=True)

    torch.save(model_instance.state_dict(), model_path)
    print(f"Model saved to: {model_path}")


if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()
    # Run the main function
    main(args)