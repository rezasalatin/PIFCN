import os
import time
import numpy as np
from glob import glob
from scipy.io import loadmat, savemat
import torch
from torch.utils.data import DataLoader, TensorDataset
from config.read_yaml import ConfigLoader
from models.AE_Res50 import AutoEncoder
import myutils
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--config-path', type=str, default='config/test_configuration.yaml',
                        help='Config path.')
    return parser.parse_args()

class ConfigManager:
    def __init__(self, config_path):
        self.config_path = config_path
        self.dataset_config = ConfigLoader(config_path, section='dataset').get_value()
        self.test_config = ConfigLoader(config_path, section='test').get_value()
        self.environment_config = ConfigLoader(config_path, section='environment').get_value()

class Tester:
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.device = self._setup_device()
        self.model = self._load_model()
        self.dataloader = self._setup_dataloader()
        self.fid_loss_fn = torch.nn.SmoothL1Loss().to(self.device)
        self.model.eval()  # Set model to evaluation mode

    def _setup_device(self):
        gpu = self.config_manager.environment_config['gpu']
        return torch.device(f"cuda:{gpu}" if torch.cuda.is_available() and gpu >= 0 else "cpu")
    
    def _load_model(self):
        model = AutoEncoder().to(self.device)
        model_path = self.config_manager.test_config['model_path']
        # Load the entire checkpoint, not just the model state dictionary
        checkpoint = torch.load(model_path, map_location=self.device)
        # Extract the model state dictionary
        model_state_dict = checkpoint['model_state']
        model.load_state_dict(model_state_dict)
        return model

    def _setup_dataloader(self):
        input_data = loadmat(self.config_manager.dataset_config['paths']['input'])['test_input']
        target_data = loadmat(self.config_manager.dataset_config['paths']['target'])['test_target']
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        target_tensor = torch.tensor(target_data, dtype=torch.float32)
        dataset = TensorDataset(input_tensor, target_tensor)
        return DataLoader(dataset, batch_size=self.config_manager.test_config['batch_size'], shuffle=False, num_workers=2, pin_memory=True)

    def evaluate(self):
        all_predictions = []  # Initialize a list to collect predictions

        for inputs, targets in self.dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            with torch.no_grad():
                preds = self.model(inputs)
                all_predictions.append(preds.cpu())  # Append predictions to the list

        # Concatenate all batch predictions into a single tensor
        all_predictions = torch.cat(all_predictions, dim=0)

        # Convert the tensor to a numpy array
        all_predictions_np = all_predictions.numpy()

        # Define the path where you want to save the predictions
        predictions_path = os.path.join(self.config_manager.test_config['results_dir'], 'all_predictions.mat')
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(predictions_path), exist_ok=True)

        # Save the predictions to a .mat file
        savemat(predictions_path, {'predictions': all_predictions_np})

        print(f'All predictions saved to {predictions_path}')

def main():
    args = get_args()
    config_manager = ConfigManager(args.config_path)
    tester = Tester(config_manager)
    tester.evaluate()

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    test_time = round(end_time - start_time)
    print(f'Testing time: {test_time} seconds.')
