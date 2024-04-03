import os
import time
import numpy as np
from glob import glob
from scipy.io import loadmat, savemat
import torch
from torch.utils.data import DataLoader, TensorDataset
from config.read_yaml import ConfigLoader
from models.AE_Res50 import AutoEncoder
from models.physics import continuity_only as res_loss_fn
import myutils
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--config-path', type=str, default='config/training_configuration.yaml',
                        help='Config path.')
    return parser.parse_args()

class ConfigManager:
    def __init__(self, config_path):
        self.config_path = config_path
        self.dataset_config = ConfigLoader(config_path, section='dataset').get_value()
        self.training_config = ConfigLoader(config_path, section='training').get_value()
        self.environment_config = ConfigLoader(config_path, section='environment').get_value()
        self.physics_config = ConfigLoader(config_path, section='physics').get_value()
        
class Trainer:
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.device = self._setup_device()
        self.set_seeds(self.config_manager.environment_config['seed'])
        self.model = AutoEncoder().to(self.device)
        self.optimizer = self._setup_optimizer()
        self.fid_loss_fn = torch.nn.SmoothL1Loss().to(self.device)
        self.scheduler = self._setup_scheduler()
        self.dataloader = self._setup_dataloader()
        self.model.train()
        self.model.apply(myutils.set_bn_eval)  # turn-off BN
        self.main_dir = self._setup_logging_directory()
        self.model_dir = os.path.join(self.main_dir, 'model')
        os.makedirs(self.model_dir, exist_ok=True)
        self.log_path = os.path.join(self.main_dir, 'training_log.txt')
        self._save_scripts()

    def _setup_device(self):
        gpu = self.config_manager.environment_config['gpu']
        return torch.device(f"cuda:{gpu}" if torch.cuda.is_available() and gpu >= 0 else "cpu")

    def set_seeds(self, seed_value):
        if seed_value < 0:
            seed_value = int(time.time())
        torch.manual_seed(seed_value)
        np.random.seed(seed_value)
        print(f"Seeds set to: {seed_value}")

    def _setup_logging_directory(self):
        if not os.path.exists('logs'):
            os.makedirs('logs')
        main_dir = os.path.join('logs', time.strftime('%Y%m%d-%H%M'))
        os.makedirs(main_dir, exist_ok=True)
        return main_dir
    
    def _save_scripts(self):
        myutils.save_scripts(self.main_dir, scripts_to_save=glob('*.*'))
        myutils.save_scripts(self.main_dir, scripts_to_save=glob('config/*.*', recursive=True))
        myutils.save_scripts(self.main_dir, scripts_to_save=glob('dataset/*.py', recursive=True))
        myutils.save_scripts(self.main_dir, scripts_to_save=glob('models/*.py', recursive=True))
        myutils.save_scripts(self.main_dir, scripts_to_save=glob('myutils/*.py', recursive=True))

    def _log_training_progress(self, epoch, avg_loss, avg_fid_loss, avg_res_loss):
        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, Avg Loss: {avg_loss:.4f}, Avg FID Loss: {avg_fid_loss:.4f}, Avg RES Loss: {avg_res_loss:.4f}")

    def _save_checkpoint(self, epoch, avg_loss):
        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'avg_loss': avg_loss
        }
        if epoch % 200 == 0:
            torch.save(checkpoint, os.path.join(self.model_dir, f'checkpoint_epoch_{epoch}.pth'))
            print(f'Saved model for epoch {epoch}')

        with open(self.log_path, 'a') as log_file:
            log_file.write(f'Epoch: {epoch}, Average Loss: {avg_loss:.5f}\n')

    def _save_predictions(self, epoch, predictions):
        savemat(os.path.join(self.model_dir, f'predictions_epoch_{epoch}.mat'), {'predictions': predictions})
        print(f'Saved predictions for epoch {epoch}')

    def _setup_optimizer(self):
        return torch.optim.AdamW(filter(lambda x: x.requires_grad, self.model.parameters()),
                                 lr=self.config_manager.training_config['learning_rate'],
                                 weight_decay=self.config_manager.training_config['regularization'])

    def _setup_scheduler(self):
        return torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.config_manager.training_config['scheduler']['step_size'],
            gamma=self.config_manager.training_config['scheduler']['gamma'])

    def _setup_dataloader(self):
        input_data = loadmat(self.config_manager.dataset_config['paths']['input'])['train_input']
        target_data = loadmat(self.config_manager.dataset_config['paths']['target'])['train_target']
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        target_tensor = torch.tensor(target_data, dtype=torch.float32)
        dataset = TensorDataset(input_tensor, target_tensor)
        return DataLoader(dataset, batch_size=self.config_manager.training_config['batch_size'], shuffle=True, drop_last=True, num_workers=2, pin_memory=True)

    def train_epoch(self, epoch):
        stats_loss = myutils.AvgMeter()
        stats_fid_loss = myutils.AvgMeter()
        stats_res_loss = myutils.AvgMeter()

        last_preds = None  # Initialize to store last batch predictions

        for iter_idx, (inputs, targets) in enumerate(self.dataloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            preds = self.model(inputs)
            self.optimizer.zero_grad()
            loss, fid_loss, res_loss = self._calculate_loss(inputs, targets, preds)
            loss.backward()
            self.optimizer.step()

            # Update the AvgMeters
            stats_loss.update(loss.item())
            stats_fid_loss.update(fid_loss.item())
            stats_res_loss.update(res_loss.item())

            if (epoch+1) % 200 == 0:
                last_preds = preds.detach().cpu().numpy()

        # Get the average losses for the epoch
        avg_loss = stats_loss.avg
        avg_fid_loss = stats_fid_loss.avg
        avg_res_loss = stats_res_loss.avg

        # Log the average losses
        self._log_training_progress(epoch + 1, avg_loss, avg_fid_loss, avg_res_loss)

        self._save_checkpoint(epoch+1, avg_loss)  # Save checkpoint with the average loss
        if last_preds is not None:
            self._save_predictions(epoch+1, last_preds)  # Save predictions for this epoch          


    def _calculate_loss(self, inputs, targets, preds):
        loss = 0.0  # Initialize loss for this batch

        # Apply mask and calculate valid losses
        mask = torch.zeros_like(targets, dtype=torch.bool)
        myutils.point_selector(mask, x_intv=2, y_intv=2, random=False, num_points=50)
        targets[~mask] = torch.nan
        
        valid_mask = ~torch.isnan(preds) & ~torch.isnan(targets)
        preds_valid = preds[valid_mask]
        targets_valid = targets[valid_mask]

        # Fidelity loss
        if self.config_manager.training_config['fidelity']:
            fid_loss = self.fid_loss_fn(preds_valid, targets_valid)
            loss += fid_loss

        # Physics-based loss
        if self.config_manager.training_config['physics']:
            batch_size = self.config_manager.training_config['batch_size']
            dx = self.config_manager.physics_config['dx']
            dy = self.config_manager.physics_config['dy']
            delta = self.config_manager.physics_config['huber_delta']
            res_losses = [res_loss_fn(inputs[i,:,:,:].squeeze(), preds[i,:,:,:].squeeze(), dx, dy, delta) for i in range(batch_size)]
            res_loss = torch.tensor(res_losses, device=self.device).mean()
            loss += res_loss

        return loss, fid_loss, res_loss

    def train(self):
        for epoch in range(self.config_manager.training_config['max_epoch']):
            self.train_epoch(epoch)
            self.scheduler.step()

def main():
    args = get_args()  # Your existing get_args function
    config_manager = ConfigManager(args.config_path)
    trainer = Trainer(config_manager)
    trainer.train()

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    training_time = round(end_time - start_time)
    print(f'Training time: {training_time} seconds.')
