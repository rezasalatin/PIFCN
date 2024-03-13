import os
import numpy as np
import yaml
from scipy.io import loadmat
from glob import glob
import torch
from torch.utils import data
import torchvision.transforms as TF
from config.read_yaml import ConfigLoader


class Training_dataset(data.Dataset):

    def __init__(self, config_path):
        
        self.dataset_path = ConfigLoader(config_path, section='dataset', key='path').get_value()
        self.output_size = ConfigLoader(config_path, section='dataset', key='output_size').get_value()
        


    def __getitem__(self, idx):

        # Load MATLAB file
        mat_path = self.dataset_files[idx]
        mat_contents = loadmat(mat_path)
        
        # Assuming the key in the MATLAB file containing the data is 'data'
        # Adjust 'data' to the actual variable name in your MATLAB files
        data = mat_contents['data']
        
        # Convert the data to a PyTorch tensor and permute to match PyTorch's shape (C, H, W)
        data_tensor = torch.tensor(data, dtype=torch.float).permute(2, 0, 1)
        
        return data_tensor