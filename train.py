import os
import time
import argparse
import numpy as np
from glob import glob
from scipy.io import loadmat, savemat
import torch
from torch.utils.data import DataLoader, TensorDataset
from config.read_yaml import ConfigLoader
from models.AE_Res50 import AutoEncoder
from models.physics import continuity_only as res_loss_fn
import myutils

class Trainer:
    def __init__(self, args):
        self.args = args
        self.load_config()
        self.setup_device()
        self.prepare_logging()
        self.load_data()
        self.prepare_model()

    def load_config(self):
        self.config_loader = ConfigLoader(self.args.config_path)
        self.conf = {k: self.config_loader.get_value(section='initialization', key=k) for k in
                     ['log', 'lr', 'seed', 'scheduler_step', 'total_epochs', 'batch_size', 'gpu']}

    def setup_device(self):
        gpu = int(self.conf['gpu'])
        self.device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() and gpu >= 0 else "cpu")
        print(f"Using device: {self.device}")

    def prepare_logging(self):
        if self.conf['log']:
            main_dir = 'logs/{}'.format(time.strftime('%Y%m%d-%H%M'))
            self.log_dir = os.path.join(main_dir, 'log')
            self.model_dir = os.path.join(main_dir, 'model')
            os.makedirs(self.log_dir, exist_ok=True)
            os.makedirs(self.model_dir, exist_ok=True)
            myutils.save_scripts(main_dir, scripts_to_save=glob('*.*'))
            myutils.save_scripts(main_dir, scripts_to_save=glob('config/*.*', recursive=True))
            myutils.save_scripts(main_dir, scripts_to_save=glob('dataset/*.py', recursive=True))
            myutils.save_scripts(main_dir, scripts_to_save=glob('models/*.py', recursive=True))
            myutils.save_scripts(main_dir, scripts_to_save=glob('myutils/*.py', recursive=True))

    def load_data(self):
        # Similar to the original main function, load data here

    def prepare_model(self):
        self.model = AutoEncoder(self.device).to(self.device)
        self.model.train()
        self.model.apply(myutils.set_bn_eval)
        optimizer_params = filter(lambda x: x.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.AdamW(optimizer_params, self.conf['lr'])
        self.fid_loss_fn = torch.nn.SmoothL1Loss().to(self.device)

    def train(self):
        # Implement the training loop here, similar to the train_model function

    def run(self):
        start_time = time.time()
        self.train()
        end_time = time.time()
        print(myutils.gct(), 'Training done.')
        print("Training time:", end_time - start_time, "seconds.")

def get_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--config-path', type=str, default='config/training_configuration.yaml',
                        help='Config path.')
    return parser.parse_args()

def main():
    args = get_args()
    trainer = Trainer(args)
    trainer.run()

if __name__ == '__main__':
    main()
