"""
Script to run predictions using a pre-trained model and visualize the results.

This script performs the following steps:
1. Parses command-line arguments to get the model path and other settings.
2. Infers model type and depth from the model's filename.
3. Loads the trained model weights.
4. Loads and preprocesses the test data.
5. Runs the model in evaluation mode to get temperature predictions.
6. Plots the predicted temperatures against the actual temperatures.
7. Saves the plot to a specified output directory.

How to run:
python your_prediction_script_name.py --model_path PhysNet_depth8_seed1.pth
"""

import argparse
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assuming 'train.py' is in the same directory or accessible in the python path
from train import prepare_data
from utils.PhysRegMLP import PhysRegMLP
from utils.PhysMLP import PhysMLP
from utils.PhysNet import PhysNet
from utils.support_functions import inverse_transform_temp

def get_model_and_params(model_type: str, depth: int, lambda_value: float = 0.5) -> tuple:
    """
    Returns the model class and its network configuration based on model type and depth.
    
    Args:
        model_type (str): The type of the model ('PhysNet' or 'PhysRegMLP').
        depth (int): The historical depth used during training.
        lambda_value (float): The lambda value used during training.

    Returns:
        tuple: (Model class, Dictionary of network parameters)
    """
    if model_type == 'PhysRegMLP':
        model_class = PhysRegMLP
        network_param = {
            'lr': 0.001, 'batch_size': 2048, 'lambda_value': lambda_value,
            'mdp_network': {'input_size': 4 + 2 * depth, 'fc': [64] * 2, 'output_size': 3, 'activation': 'tanh', 'dropout_rate': 0.01}
        }
    elif model_type == 'PhysNet':
        model_class = PhysNet
        network_param = {
            'lr': 0.001, 'batch_size': 2048, 'lambda_value': lambda_value,
            'encoding_network': {'input_size': 2 * depth, 'fc': [24] * 1, 'output_size': 1, 'activation': 'tanh', 'dropout_rate': 0.01},
            'mdp_network': {'input_size': 5, 'fc': [128] * 1, 'output_size': 2, 'activation': 'tanh', 'dropout_rate': 0.05}
        }    
    elif model_type == 'PhysMLP':
        model_class = PhysMLP
        network_param = {
            'lr': 0.001,
            'batch_size': 2048,
            'lambda_value': lambda_value,
            'mdp_network': {
                'input_size': 4 + 2 * depth,
                'fc': [64] * 4,
                'output_size': 3,
                'activation': ['relu', 'tanh', 'relu', 'tanh'],
                'dropout_rate': 0.01
            }
        }
    else:
        raise TypeError(f"Unsupported model type: {model_type}")
    return model_class, network_param

def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for the prediction script.
    """
    parser = argparse.ArgumentParser(description="Run predictions with a trained model and generate a comparison plot.")
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model file (.pth). Example: "PhysNet_depth8_seed1.pth"')
    parser.add_argument('--test_data_path', type=str, default='data/Training_data.csv',
                        help='Path to the test data CSV file (default: data/Test_data.csv)')
    parser.add_argument('--output_dir', type=str, default='predictions',
                        help='Directory to save the output plot (default: predictions)')

    return parser.parse_args()

def main(args: argparse.Namespace):
    """
    Main function to run the prediction and plotting pipeline.
    """
    # 1. Parse model details from the filename
    model_filename = os.path.basename(args.model_path)
    try:
        parts = model_filename.replace('.pth', '').split('_')
        model_type = parts[0]
        depth = int(parts[1].replace('depth', ''))
    except (IndexError, ValueError):
        print("Error: Model filename is not in the expected format.")
        print("Expected format: 'ModelType_depthX_seedY.pth' (e.g., 'PhysNet_depth8_seed1.pth')")
        return

    print(f"Model Type: {model_type}, Depth: {depth}")

    # 2. Initialize and load the model
    model_class, network_param = get_model_and_params(model_type, depth)
    model_instance = model_class(network_param)
    
    try:
        model_instance.load_state_dict(torch.load(args.model_path))
    except FileNotFoundError:
        print(f"Error: Model file not found at '{args.model_path}'")
        return

    model_instance.eval() 
    print(f"Successfully loaded model from {args.model_path}")

    # 3. Load and prepare test data
    try:
        test_data_df = pd.read_csv(args.test_data_path)
    except FileNotFoundError:
        print(f"Error: Test data file not found at '{args.test_data_path}'")
        return
        
    test_data_main = test_data_df.to_numpy()

    training_data_index_selection = tuple(
        [0, 1, *list(np.arange(4, 4 + depth)), *list(np.arange(4 + 24, 4 + 24 + depth)), 2, -1])
    
    input_data_dict = {
        'x_agg_k': test_data_main[:, training_data_index_selection],
        'label_k': test_data_main[:, (3, -2)]
    }
    
    # Correctly call prepare_data with depth argument
    test_data_dict = prepare_data(input_data_dict, depth)
    test_inputs = torch.tensor(test_data_dict['x_agg_k'], dtype=torch.float32)
    real_temperatures = inverse_transform_temp(test_data_dict['label_k'][:, 0])

    # 4. Run prediction
    print("Running prediction...")
    with torch.no_grad():
        predictions_scaled, _ = model_instance.forward(test_inputs)
        predicted_temperatures = inverse_transform_temp(predictions_scaled[:, 0].numpy())
    print("Prediction finished.")

    mse = np.mean((real_temperatures - predicted_temperatures) ** 2)
    print(f"Mean Squared Error (MSE): {mse:.4f}")

    # 5. Plot and save results
    plt.figure(figsize=(16, 8))
    plt.plot(real_temperatures, label='Actual Temperature', color='blue', linewidth=2)
    plt.plot(predicted_temperatures, label='Predicted Temperature', color='red', linestyle='--', linewidth=2)
    plt.title(f'Prediction Results for {model_filename}')
    plt.xlabel('Time Step (30 min intervals)')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(True)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    output_image_filename = f"prediction_results_{model_filename.replace('.pth', '.png')}"
    output_image_path = os.path.join(args.output_dir, output_image_filename)
    
    plt.savefig(output_image_path)
    plt.close()
    
    print(f"Plot saved to: {output_image_path}")

if __name__ == '__main__':
    args = parse_args()
    main(args)