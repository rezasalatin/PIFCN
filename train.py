# Standard Library Imports
import os
import time
import argparse
from glob import glob

# Third-Party Imports
import numpy as np
from scipy.io import loadmat, savemat
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

# Local Application/Library Specific Imports
from config.read_yaml import ConfigLoader
from models.vae_res101 import VAE
from models.physics import simplified_swe as res_loss_fn
import myutils
from myutils.augmentations import augment_flow

def get_args():
    parser = argparse.ArgumentParser(description='Train a VAE model.')
    parser.add_argument('--config-path', type=str, default='config/train_config.yaml', help='Path to the configuration file.')
    return parser.parse_args()

class ConfigManager:
    def __init__(self, config_path):
        self.config_path = config_path
        self.dataset_config = ConfigLoader(config_path, section='dataset').get_value()
        self.training_config = ConfigLoader(config_path, section='training').get_value()
        self.environment_config = ConfigLoader(config_path, section='environment').get_value()
        self.physics_config = ConfigLoader(config_path, section='physics').get_value()
         # Add default values for total variation loss configuration
        self.training_config['total_variation'] = self.training_config.get('total_variation', False)
        self.training_config['tv_loss_coef'] = self.training_config.get('tv_loss_coef', 0.0001)

class Trainer:
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.device = self._setup_device()
        self.set_seeds(self.config_manager.environment_config['seed'])
        self.model = self._setup_model()
        self.optimizer = self._setup_optimizer()
        self.fid_loss_fn = torch.nn.SmoothL1Loss().to(self.device)
        self.scheduler = self._setup_scheduler()
        self.dataloader = self._setup_dataloader()
        self._prepare_for_training()

    def _setup_device(self):
        gpu = self.config_manager.environment_config['gpu']
        return torch.device(f"cuda:{gpu}" if torch.cuda.is_available() and gpu >= 0 else "cpu")

    def set_seeds(self, seed_value):
        if seed_value < 0:
            seed_value = int(time.time())
        torch.manual_seed(seed_value)
        np.random.seed(seed_value)
        print(f"Seeds set to: {seed_value}")

    def _setup_model(self):
        input_channels = self.config_manager.dataset_config['channels']['input']
        output_channels = self.config_manager.dataset_config['channels']['output']
        model = VAE(input_channels=input_channels, output_channels=output_channels, device=self.device).to(self.device)
        model.train()
        model.apply(myutils.set_bn_eval)
        return model

    def _setup_optimizer(self):
        return torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),
                                 lr=self.config_manager.training_config['learning_rate'],
                                 weight_decay=self.config_manager.training_config['regularization'])

    def _setup_scheduler(self):
        scheduler_config = self.config_manager.training_config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'StepLR')
        
        if scheduler_type == 'CyclicLR':
            return torch.optim.lr_scheduler.CyclicLR(
                self.optimizer,
                base_lr=scheduler_config.get('base_lr', 1e-6),
                max_lr=scheduler_config.get('max_lr', 1e-3),
                step_size_up=scheduler_config.get('step_size_up', 200),
                mode=scheduler_config.get('mode', 'triangular2'),
                cycle_momentum=scheduler_config.get('cycle_momentum', False)
            )
        elif scheduler_type == 'StepLR':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 100),
                gamma=scheduler_config.get('gamma', 0.9)
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    def _setup_dataloader(self):
        input_data = loadmat(self.config_manager.dataset_config['paths']['input'])['train_input']
        target_data = loadmat(self.config_manager.dataset_config['paths']['target'])['train_target']
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        target_tensor = torch.tensor(target_data, dtype=torch.float32)
        dataset = TensorDataset(input_tensor, target_tensor)
        return DataLoader(dataset, batch_size=self.config_manager.training_config['batch_size'], shuffle=True, drop_last=True, num_workers=2, pin_memory=True)

    def _prepare_for_training(self):
        self.main_dir = self._setup_logging_directory()
        self.model_dir = os.path.join(self.main_dir, 'model')
        os.makedirs(self.model_dir, exist_ok=True)
        self.log_path = os.path.join(self.main_dir, 'training_log.txt')
        self._save_scripts()
        self.best_loss = float('inf')

    def _setup_logging_directory(self):
        if not os.path.exists('logs'):
            os.makedirs('logs')
        main_dir = os.path.join('logs', time.strftime('%Y%m%d-%H%M'))
        os.makedirs(main_dir, exist_ok=True)
        return main_dir

    def _save_scripts(self):
        scripts_to_save = glob('*.*') + glob('config/*.*', recursive=True) + glob('dataset/*.py', recursive=True) + glob('models/*.py', recursive=True) + glob('myutils/*.py', recursive=True)
        myutils.save_scripts(self.main_dir, scripts_to_save=scripts_to_save)

    def _log_training_progress(self, epoch, avg_loss, avg_fid_loss, avg_res_loss, avg_tv_loss, lr):
        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, Loss: {avg_loss:.4f}, FID Loss: {avg_fid_loss:.4f}, RES Loss: {avg_res_loss:.4f}, TV Loss: {avg_tv_loss:.4f}, LR: {lr:.6f}")

        with open(self.log_path, 'a') as log_file:
            log_file.write(f"Epoch: {epoch}, Loss: {avg_loss:.6f}, FID Loss: {avg_fid_loss:.6f}, RES Loss: {avg_res_loss:.6f}, TV Loss: {avg_tv_loss:.6f}, LR: {lr:.6f}\n")

    def _save_checkpoint(self, epoch, avg_loss, avg_fid_loss, avg_res_loss, avg_tv_loss):
        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'avg_loss': avg_loss
        }
        if epoch % 10000 == 0:
            torch.save(checkpoint, os.path.join(self.model_dir, f'checkpoint_epoch_{epoch}.pth'))
            print(f'Saved model checkpoint for epoch {epoch}')

    def _save_predictions(self, epoch, predictions):
        savemat(os.path.join(self.model_dir, f'predictions_epoch_{epoch}.mat'), {'predictions': predictions})
        print(f'Saved predictions for epoch {epoch}')

    def _save_best_model(self, epoch, avg_loss, avg_fid_loss, avg_res_loss, avg_tv_loss, lr):
        improvement_threshold = 0.01
        if avg_loss < self.best_loss * (1 - improvement_threshold):
            self.best_loss = avg_loss
            best_model_path = os.path.join(self.model_dir, 'best.pth')
            torch.save({
                'epoch': epoch,
                'model_state': self.model.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'best_loss': self.best_loss
            }, best_model_path)
            print(f"Epoch: {epoch}, Loss: {avg_loss:.4f}, FID: {avg_fid_loss:.4f}, RES: {avg_res_loss:.4f}, TV: {avg_tv_loss:.4f}, LR: {lr:.8f} - Best Model Saved")

    def _calculate_loss(self, inputs, targets, preds, mu, logvar):
        loss = 0.0
        fid_loss = torch.tensor(float('nan'), device=self.device)
        res_loss = torch.tensor(float('nan'), device=self.device)
        tv_loss = torch.tensor(float('nan'), device=self.device)

        mask = torch.isnan(targets) 
        mask = myutils.point_selector(mask, x_intv=1, y_intv=1)
        preds_valid = preds[~mask]
        targets_valid = targets[~mask]

        if self.config_manager.training_config['fidelity']:
            fid_loss_coef = self.config_manager.training_config['fid_loss_coef']
            BCE = self.fid_loss_fn(preds_valid, targets_valid)
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            fid_loss = BCE + KLD
            fid_loss = fid_loss_coef * fid_loss
            loss += fid_loss_coef * fid_loss

        if self.config_manager.training_config['physics']:
            phy_loss_coef = self.config_manager.training_config['phy_loss_coef']
            batch_size = self.config_manager.training_config['batch_size']
            dx = self.config_manager.physics_config['dx']
            dy = self.config_manager.physics_config['dy']
            delta = self.config_manager.physics_config['huber_delta']
            res_losses = [res_loss_fn(inputs[i,:,:,:], preds[i,:,:,:], dx, dy, delta) for i in range(batch_size)]
            res_loss = torch.tensor(res_losses, device=self.device).mean()
            res_loss = phy_loss_coef * res_loss
            loss += res_loss

        if self.config_manager.training_config.get('total_variation', False):
            tv_loss_coef = self.config_manager.training_config['tv_loss_coef']
            tv_loss = self.total_variation_loss(preds)
            tv_loss = tv_loss_coef * tv_loss
            loss += tv_loss

        return loss, fid_loss, res_loss, tv_loss
    
    def total_variation_loss(self, img):
        batch_size, c, h, w = img.size()
        tv_h = torch.pow(img[:,:,1:,:] - img[:,:,:-1,:], 2).sum()
        tv_w = torch.pow(img[:,:,:,1:] - img[:,:,:,:-1], 2).sum()
        return (tv_h + tv_w) / (batch_size * c * h * w)

    def train_epoch(self, epoch):
        stats_loss = myutils.AvgMeter()
        stats_fid_loss = myutils.AvgMeter()
        stats_res_loss = myutils.AvgMeter()
        stats_tv_loss = myutils.AvgMeter()
        last_preds = None

        for iter_idx, (inputs, targets) in enumerate(self.dataloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Apply augmentation only during training
            if self.config_manager.training_config.get("augmentation", True):
                U, V, h = inputs[:, 0, :, :], inputs[:, 1, :, :], targets[:, 0, :, :]
                U, V, h = augment_flow(U, V, h, output_size=(inputs.shape[2], inputs.shape[3]))
                inputs = torch.stack((U, V), dim=1)
                targets = h.unsqueeze(1)

            preds, mu, logvar = self.model(inputs)

            self.optimizer.zero_grad()
            loss, fid_loss, res_loss, tv_loss = self._calculate_loss(inputs, targets, preds, mu, logvar)
            loss.backward()
            self.optimizer.step()
            
            # Step the scheduler after every batch if it's CyclicLR
            if isinstance(self.scheduler, torch.optim.lr_scheduler.CyclicLR):
                self.scheduler.step()

            stats_loss.update(loss.item())
            stats_fid_loss.update(fid_loss.item())
            stats_res_loss.update(res_loss.item())
            stats_tv_loss.update(tv_loss.item())

            if (epoch + 1) % 10000 == 0:
                last_preds = preds.detach().cpu().numpy()

        avg_loss = stats_loss.avg
        avg_fid_loss = stats_fid_loss.avg
        avg_res_loss = stats_res_loss.avg
        avg_tv_loss = stats_tv_loss.avg

        # Get current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']

        self._log_training_progress(epoch + 1, avg_loss, avg_fid_loss, avg_res_loss, avg_tv_loss, current_lr)
        self._save_checkpoint(epoch + 1, avg_loss, avg_fid_loss, avg_res_loss, avg_tv_loss)
        if last_preds is not None:
            self._save_predictions(epoch + 1, last_preds)
        self._save_best_model(epoch + 1, avg_loss, avg_fid_loss, avg_res_loss, avg_tv_loss, current_lr)

        # Step the scheduler after each epoch if it's StepLR
        if isinstance(self.scheduler, torch.optim.lr_scheduler.StepLR):
            self.scheduler.step()

    def train(self):
        start_time = time.time()
        num_epochs = self.config_manager.training_config['max_epoch']
        for epoch in range(num_epochs):
            self.train_epoch(epoch)
        end_time = time.time()
        training_time = round(end_time - start_time)
        with open(self.log_path, 'a') as log_file:
            log_file.write(f"Total Training Time: {training_time} seconds\n")
        print(f'Total Training Time: {training_time} seconds')

def main():
    args = get_args()
    config_manager = ConfigManager(args.config_path)
    trainer = Trainer(config_manager)
    trainer.train()

if __name__ == '__main__':
    main()
