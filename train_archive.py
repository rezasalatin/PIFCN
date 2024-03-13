import os
import time
import argparse
import numpy as np
from glob import glob
from scipy.io import loadmat, savemat
#from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, TensorDataset

from config.read_yaml import ConfigLoader
#from dataset.DatSet import Training_dataset
from models.AE_Res50 import AutoEncoder
from models.physics import continuity_only as res_loss_fn
import myutils

#from torchviz import make_dot


def get_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--config-path', type=str, default='config/training_configuration.yaml',
                        help='Config path.')
    return parser.parse_args()


def train_model(device, epoch, model, dataloader, fid_loss_fn, optimizer, conf_batch):

    stats = myutils.AvgMeter()

    for iter_idx, sample in enumerate(dataloader):
        
        inputs = sample[0].to(device)
        targets = sample[1].to(device)
        preds = model(inputs)
        
        #graph = make_dot(prediction, params=dict(model.named_parameters()))
        #graph.render('model_graph', format='png')
        #graph

        optimizer.zero_grad()
        
        # Create a mask with the same shape as targets, initially set to False
        mask = torch.zeros_like(targets, dtype=torch.bool)
        # Set the boundary indices of the last two dimensions to True
        mask[:, :, ::32, ::8] = True
        #mask[:, :, 0, ::4] = True   # First row in 3rd dimension
        #mask[:, :, 128, ::4] = True # Middle row in 3rd dimension
        #mask[:, :, -1, ::4] = True  # Last row in 3rd dimension
        #mask[:, :, :, 0] = True     # First column in 4tt dimension
        #mask[:, :, :, -1] = True    # Last column in 4th dimension
        targets[~mask] = torch.nan

        
        valid_mask = ~torch.isnan(preds) & ~torch.isnan(targets)
        # Apply the mask to both predictions and targets to remove NaNs
        preds_valid = preds[valid_mask]
        targets_valid = targets[valid_mask]
        fid_loss = fid_loss_fn(preds_valid, targets_valid)
        
        loss_tmp = 0
        for i in range(conf_batch):
            res_in = inputs[i,:,:,:].squeeze()
            pre_in = preds[i,:,:,:].squeeze()
            loss_tmp += res_loss_fn(res_in, pre_in)
            
        res_loss = torch.mean(loss_tmp)
        
        loss = res_loss + fid_loss
        
        loss.backward()
        optimizer.step()

        stats.update(loss.item())

        # Print only when epoch % 10 == 0
        if epoch % 10 == 0:
            print(f'Epoch [{epoch}], Iteration [{iter_idx + 1}/{len(dataloader)}], Loss: {loss.item():.5f}')
            
    # If epoch is 500, save the last prediction to a .mat file
    if epoch % 200 == 0:
        prediction_np = preds.cpu().detach().numpy()
        savemat(f'prediction_epoch_{epoch}.mat', {'prediction': prediction_np})

    return stats.avg



def main():  
    
    # get args
    args = get_args()
    print(myutils.gct(), f'Args = {args}')
    
    conf_log = ConfigLoader(args.config_path, section='initialization', key='log').get_value()
    conf_lr = ConfigLoader(args.config_path, section='initialization', key='lr').get_value()
    conf_seed = ConfigLoader(args.config_path, section='initialization', key='seed').get_value()
    conf_scheduler_step = ConfigLoader(args.config_path, section='initialization', key='scheduler_step').get_value()
    conf_epoch = ConfigLoader(args.config_path, section='initialization', key='total_epochs').get_value()
    conf_batch = ConfigLoader(args.config_path, section='initialization', key='batch_size').get_value()
    
    # Ensure 'device' is defined outside of main if not already done
    gpu = ConfigLoader(args.config_path, section='initialization', key='gpu').get_value()
    gpu = int(gpu)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() and gpu >= 0 else "cpu")
    print(f"Using device: {device}")

    if conf_log:
        if not os.path.exists('logs'):
            os.makedirs('logs')

        main_dir = 'logs/{}'.format(time.strftime('%Y%m%d-%H%M'))
        log_dir = os.path.join(main_dir, 'log')
        model_dir = os.path.join(main_dir, 'model')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_path = os.path.join(log_dir, 'training_log.txt')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        myutils.save_scripts(main_dir, scripts_to_save=glob('*.*'))
        myutils.save_scripts(main_dir, scripts_to_save=glob('config/*.*', recursive=True))
        myutils.save_scripts(main_dir, scripts_to_save=glob('dataset/*.py', recursive=True))
        myutils.save_scripts(main_dir, scripts_to_save=glob('models/*.py', recursive=True))
        myutils.save_scripts(main_dir, scripts_to_save=glob('myutils/*.py', recursive=True))
    
    # (N, C, H, W) format is expected
    data_path = ConfigLoader(args.config_path, section='dataset', key='input_path').get_value()
    input_data = loadmat(data_path)
    input_data = input_data['input_data']
    input_tensor = torch.tensor(input_data, dtype=torch.float32)

    data_path = ConfigLoader(args.config_path, section='dataset', key='target_path').get_value()
    target_data = loadmat(data_path)
    target_data = target_data['h']
    target_tensor = torch.tensor(target_data, dtype=torch.float32)
    
    
    # TensorDataset and DataLoader expect data to be in (N, C, H, W) format
    dataset = TensorDataset(input_tensor,target_tensor)
    dataloader = DataLoader(dataset, batch_size=conf_batch, shuffle=True, num_workers=2, pin_memory=True)
    
    model = AutoEncoder(device)
    model = model.to(device)
    model.train()
    model.apply(myutils.set_bn_eval)  # turn-off BN

    params = model.parameters()
    optimizer = torch.optim.AdamW(filter(lambda x: x.requires_grad, params), conf_lr)

    start_epoch = 0
    best_loss = 100000000

    if conf_seed < 0:
        seed = int(time.time())
    else:
        seed = conf_seed

    torch.manual_seed(seed)
    np.random.seed(seed)

    fid_loss_fn = torch.nn.SmoothL1Loss().to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=conf_scheduler_step, gamma=0.5, last_epoch=start_epoch - 1)

    for epoch in range(start_epoch, conf_epoch):

        lr = scheduler.get_last_lr()[0]
        
        if epoch % 10 == 0:
            print('')
            print(myutils.gct(), f'Epoch: {epoch} lr: {lr}')

        loss = train_model(device, epoch, model, dataloader, fid_loss_fn, optimizer, conf_batch)
        if conf_log:

            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': loss,
                'seed': seed,
            }

            checkpoint_path = f'{model_dir}/final.pth'
            torch.save(checkpoint, checkpoint_path)

            if best_loss > loss:
                best_loss = loss

                # don't save model very often for memory efficiency
                # checkpoint_path = f'{model_path}/epoch_{epoch:03d}_loss_{loss:.03f}.pth'
                #torch.save(checkpoint, checkpoint_path)

                checkpoint_path = f'{model_dir}/best.pth'
                torch.save(checkpoint, checkpoint_path)
            # Check if the file exists and is empty; if so, write the header
            if epoch == 0:
                with open(log_path, 'w') as log_file:
                    log_file.write('Epoch, Loss\n')
                    
            # Create a formatted string for the log values
            log_values = f'{epoch}, {loss:.5e}\n'
            # Write the log values to a file
            with open(log_path, 'a') as log_file:
                log_file.write(log_values)

        scheduler.step()


if __name__ == '__main__':

    start_time = time.time()  # Record start time
    main()  # Assuming main() is your training function
    end_time = time.time()  # Record end time
    training_time = end_time - start_time  # Calculate training time

    print(myutils.gct(), 'Training done.')
    print("Training time:", training_time, "seconds.")
