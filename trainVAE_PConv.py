import os
import time
import numpy as np
from glob import glob
from scipy.io import loadmat, savemat
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from config.read_yaml import ConfigLoader
from models.VAE_Res50_PConv import VAE
from models.physics import continuity as res_loss_fn
from models.loss_calculator_pconv import LossCalculator
import myutils
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--config-path', type=str, default='config/training_configuration_pconv.yaml',
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
        input_channels = self.config_manager.dataset_config['channels']['input']
        output_channels = self.config_manager.dataset_config['channels']['output']
        self.model = VAE(input_channels=input_channels, output_channels=output_channels).to(self.device)
        self.optimizer = self._setup_optimizer()
        self.fid_loss_fn = torch.nn.SmoothL1Loss().to(self.device)
        self.scheduler = self._setup_scheduler()
        self.dataloader = self._setup_dataloader()
        self.model.train()
        self.model.apply(myutils.set_bn_eval)
        self.main_dir = self._setup_logging_directory()
        self.model_dir = os.path.join(self.main_dir, 'model')
        os.makedirs(self.model_dir, exist_ok=True)
        self.log_path = os.path.join(self.main_dir, 'training_log.txt')
        self._save_scripts()
        self.best_loss = float('inf')  # Initialize best loss as infinity
        self.loss_calculator = LossCalculator(self.device)

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
    
    def _save_best_model(self, epoch, avg_loss, avg_fid_loss, avg_res_loss):
        improvement_threshold = 0.01  # 1% improvement
        if avg_loss < self.best_loss * (1 - improvement_threshold):
            self.best_loss = avg_loss
            best_model_path = os.path.join(self.model_dir, 'best.pth')
            torch.save({
                'epoch': epoch,
                'model_state': self.model.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'best_loss': self.best_loss
            }, best_model_path)
            print(f"Ep: {epoch}, Loss: {avg_loss:.4f}, FID Loss: {avg_fid_loss:.4f}, RES Loss: {avg_res_loss:.4f}, -Best Model")

    def _save_scripts(self):
        myutils.save_scripts(self.main_dir, scripts_to_save=glob('*.*'))
        myutils.save_scripts(self.main_dir, scripts_to_save=glob('config/*.*', recursive=True))
        myutils.save_scripts(self.main_dir, scripts_to_save=glob('dataset/*.py', recursive=True))
        myutils.save_scripts(self.main_dir, scripts_to_save=glob('models/*.py', recursive=True))
        myutils.save_scripts(self.main_dir, scripts_to_save=glob('myutils/*.py', recursive=True))

    def _log_training_progress(self, epoch, avg_loss, avg_fid_loss, avg_res_loss):
        if epoch % 10 == 0:
            print(f"Ep: {epoch}, Loss: {avg_loss:.4f}, FID Loss: {avg_fid_loss:.4f}, RES Loss: {avg_res_loss:.4f}")

    def _save_checkpoint(self, epoch, avg_loss, avg_fid_loss, avg_res_loss):
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
            log_file.write(f"Ep: {epoch}, Loss: {avg_loss:.6f}, FID Loss: {avg_fid_loss:.6f}, RES Loss: {avg_res_loss:.6f}\n")

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

            ### PConv
            # Create a mask where NaNs are 0 and non-NaNs are 1
            masks = torch.isnan(inputs)
            masks = ~masks  # Invert mask: True where x is valid, False where x is NaN
            masks = masks.float()  # Convert boolean mask to float

            preds, mu, logvar = self.model(inputs, masks)
            
            self.optimizer.zero_grad()
            loss_pconv = self.loss_calculator.combined_loss(targets, preds, masks)
            loss, fid_loss, res_loss = self._calculate_loss(targets, preds, mu, logvar)
            loss.backward()
            self.optimizer.step()

            # Update the AvgMeters
            stats_loss.update(loss.item())
            if self.config_manager.training_config['fidelity']:
                stats_fid_loss.update(fid_loss.item())
            if self.config_manager.training_config['physics']:
                stats_res_loss.update(res_loss.item())

            if (epoch+1) % 200 == 0:
                last_preds = preds.detach().cpu().numpy()

        # Get the average losses for the epoch
        avg_loss = stats_loss.avg
        avg_fid_loss = stats_fid_loss.avg
        avg_res_loss = stats_res_loss.avg

        # Log the average losses
        self._log_training_progress(epoch + 1, avg_loss, avg_fid_loss, avg_res_loss)

        self._save_checkpoint(epoch+1, avg_loss, avg_fid_loss, avg_res_loss)  # Save checkpoint with the average loss
        if last_preds is not None:
            self._save_predictions(epoch+1, last_preds)  # Save predictions for this epoch          
        self._save_best_model(epoch+1, avg_loss, avg_fid_loss, avg_res_loss)  # Check and save the best model if needed


    def _calculate_loss(self, targets, preds, mu, logvar):
        loss = 0.0
        fid_loss = 0.0
        res_loss = 0.0

        # Apply mask and calculate valid losses
        mask = torch.zeros_like(targets, dtype=torch.bool)
        myutils.point_selector(mask, x_intv=1, y_intv=1)
        targets[mask] = torch.nan
        
        valid_mask = ~torch.isnan(preds) & ~torch.isnan(targets)
        preds_valid = preds[valid_mask]
        targets_valid = targets[valid_mask]

        # Fidelity loss
        if self.config_manager.training_config['fidelity']:
            fid_loss_coef = self.config_manager.training_config['fid_loss_coef']
            # Binary Cross-Entropy Loss (BCE)
            BCE = self.fid_loss_fn(preds_valid, targets_valid)
            # KL divergence
            KLD = -0.5 * sum([torch.sum(1 + lv - m.pow(2) - lv.exp()) for m, lv in zip(mu, logvar)])
            fid_loss = BCE + KLD
            loss += fid_loss_coef*fid_loss

        # Physics-based loss
        if self.config_manager.training_config['physics']:
            phy_loss_coef = self.config_manager.training_config['phy_loss_coef']
            batch_size = self.config_manager.training_config['batch_size']
            dx = self.config_manager.physics_config['dx']
            dy = self.config_manager.physics_config['dy']
            delta = self.config_manager.physics_config['huber_delta']
            res_losses = [res_loss_fn(preds[i,:,:,:].squeeze(), dx, dy, delta) for i in range(batch_size)]
            res_loss = torch.tensor(res_losses, device=self.device).mean()
            res_loss = res_loss
            loss += phy_loss_coef*res_loss

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
