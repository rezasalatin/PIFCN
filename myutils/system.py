import time
import os
import shutil
import numpy as np
import torch
import random

class AvgMeter(object):

    def __init__(self, window=-1):
        self.window = window
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        self.max = -np.Inf

        if self.window > 0:
            self.val_arr = np.zeros(self.window)
            self.arr_idx = 0

    def update(self, val, n=1):

        self.cnt += n
        self.max = max(self.max, val)

        if self.window > 0:
            self.val_arr[self.arr_idx] = val
            self.arr_idx = (self.arr_idx + 1) % self.window
            self.avg = self.val_arr.mean()
        else:
            self.sum += val * n
            self.avg = self.sum / self.cnt


def gct(f='l'):
    '''
    get current time
    :param f: 'l' for log, 'f' for file name
    :return: formatted time
    '''
    if f == 'l':
        return time.strftime('%m/%d %H:%M:%S', time.localtime(time.time()))
    elif f == 'f':
        return time.strftime('%m_%d_%H_%M', time.localtime(time.time()))


def save_scripts(path, scripts_to_save=None):
    if not os.path.exists(os.path.join(path, 'scripts')):
        os.makedirs(os.path.join(path, 'scripts'))

    if scripts_to_save is not None:
        for script in scripts_to_save:
            dst_path = os.path.join(path, 'scripts', script)
            try:
                shutil.copy(script, dst_path)
            except IOError:
                os.makedirs(os.path.dirname(dst_path))
                shutil.copy(script, dst_path)


def print_mem(info=None):
    if info:
        print(info, end=' ')
    mem_allocated = round(torch.cuda.memory_allocated() / 1048576)
    mem_cached = round(torch.cuda.memory_cached() / 1048576)
    print(f'Mem allocated: {mem_allocated}MB, Mem cached: {mem_cached}MB')


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def initialize_seeds(seed_value=42):
    """Initialize random seeds to make experiments reproducible."""
    random.seed(seed_value)  # Python random module.
    np.random.seed(seed_value)  # Numpy module.
    torch.manual_seed(seed_value)  # PyTorch for CPU operations.
    
    # If you are using CUDA, you should also set the seed for it to ensure reproducibility.
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # For multiGPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logging_and_directories():
    main_dir = os.path.join('logs', time.strftime('%Y%m%d-%H%M'))
    log_dir = os.path.join(main_dir, 'log')
    model_dir = os.path.join(main_dir, 'model')

    for directory in [log_dir, model_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    log_path = os.path.join(log_dir, 'training_log.txt')
    return main_dir, log_dir, model_dir, log_path

def save_scripts_in_directories(directories, destination_dir):
    """
    Copies files from specified directories to a destination directory, preserving the directory structure.

    :param directories: List of directory paths to copy files from.
    :param destination_dir: Destination directory path where files will be copied.
    """
    for dir_path in directories:
        for root, _, files in os.walk(dir_path):
            for file in files:
                # Construct the path to the file to be copied
                file_path = os.path.join(root, file)
                # Create a similar directory structure in the destination directory
                relative_path = os.path.relpath(root, dir_path)
                destination_path = os.path.join(destination_dir, relative_path)
                
                # Ensure the destination directory exists
                if not os.path.exists(destination_path):
                    os.makedirs(destination_path)
                
                # Copy the file
                shutil.copy(file_path, destination_path)
                

def point_selector(mask, x_intv=8, y_intv=32, random=False, num_points=50, seed=123):

    # Set a constant seed for reproducibility
    np.random.seed(seed)
    _, _, H, W = mask.shape
    
    if not random:
        # Regular pattern selection, including boundary points
        mask[:, :, -1, ::x_intv] = True
        mask[:, :, ::y_intv, -1] = True
        mask[:, :, ::y_intv, ::x_intv] = True
    else:
        # Set boundary points
        mask[:, :, 0, ::x_intv] = True
        mask[:, :, -1, ::x_intv] = True
        mask[:, :, ::y_intv, 0] = True
        mask[:, :, ::y_intv, -1] = True
        
        # Generate random indices for the selected number of points within the interior
        random_rows = np.random.randint(1, H - 1, num_points)
        random_cols = np.random.randint(1, W - 1, num_points)

        # Ensure random points do not overlap with boundary points
        for i in range(num_points):
            mask[:, :, random_rows[i], random_cols[i]] = True

    return mask

def quadratic_extrapolate_pad(input_tensor, pad_width):
    # Assuming input_tensor is [Batch, Channels, Height, Width]
    device = input_tensor.device  # Infer the device from the input tensor

    # Prepare tensors for quadratic fitting, ensuring they're on the correct device
    x = torch.tensor([-2, -1, 0], dtype=torch.float32, device=device)
    A = torch.stack([x**2, x, torch.ones_like(x)], dim=1)

    # Left edge extrapolation
    left_values = input_tensor[:, :, :, :3]
    left_coeffs = torch.linalg.lstsq(A, left_values.permute(3, 0, 1, 2)).solution
    left_x = torch.arange(-pad_width, 0, dtype=torch.float32, device=device)
    left_pad = (left_coeffs[0] * left_x**2 + left_coeffs[1] * left_x + left_coeffs[2]).permute(1, 2, 3, 0)

    # Right edge extrapolation
    right_values = input_tensor[:, :, :, -3:]
    right_coeffs = torch.linalg.lstsq(A, right_values.permute(3, 0, 1, 2)).solution
    right_x = torch.arange(1, pad_width + 1, dtype=torch.float32, device=device)
    right_pad = (right_coeffs[0] * right_x**2 + right_coeffs[1] * right_x + right_coeffs[2]).permute(1, 2, 3, 0)

    # Apply extrapolated padding
    padded_tensor = torch.cat([left_pad, input_tensor, right_pad], dim=3)
    
    return padded_tensor