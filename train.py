import os
import time
import argparse
import numpy as np
from tqdm import tqdm
from glob import glob
from scipy.io import loadmat, savemat

import torch
from torch.utils.data import DataLoader, TensorDataset

from dataset.DatSet import Training_dataset
from models.AE import AutoEncoder
from models.physics import continuity_only as res_loss_fn
import myutils

from torchviz import make_dot


def get_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU card id.')
    parser.add_argument('--dataset', type=str, default='./input_data.mat', required=False,
                        help='Dataset folder.')
    parser.add_argument('--seed', type=int, default=-1,
                        help='Random seed.')
    parser.add_argument('--log', action='store_true',
                        help='Save the training results.')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate, default 1e-5.')
    parser.add_argument('--lu', type=float, default=0.5,
                        help='Regularization factor, default 0.5.')
    parser.add_argument('--resume', type=str,
                        help='Path to the checkpoint (default: none)')
    parser.add_argument('--new', action='store_true',
                        help='Train the model from the begining.')
    parser.add_argument('--scheduler-step', type=int, default=100,
                        help='Scheduler step size. Default 100.')
    parser.add_argument('--total-epochs', type=int, default=501,
                        help='Total running epochs. Default 10000.')
    parser.add_argument('--budget', type=int, default=300000,
                        help='Max number of features that feature bank can store. Default: 300000')

    return parser.parse_args()


def train_model(device, epoch, model, dataloader, fid_loss_fn, optimizer, target_frame):

    stats = myutils.AvgMeter()

    for iter_idx, sample in enumerate(dataloader):
        
        frames = sample[0].to(device)

        target_frame = target_frame.reshape(1, 1, 256, 64).to(device)

        prediction = model(frames)
        
        #graph = make_dot(prediction, params=dict(model.named_parameters()))
        #graph.render('model_graph', format='png')
        #graph

        optimizer.zero_grad()
        
        valid_mask = ~torch.isnan(prediction) & ~torch.isnan(target_frame)
        # Apply the mask to both predictions and targets to remove NaNs
        predictions_nonan = prediction[valid_mask]
        targets_nonan = target_frame[valid_mask]
        fid_loss = fid_loss_fn(predictions_nonan, targets_nonan)
        
        pred_temp = prediction.squeeze(0).squeeze(0)

        X = frames[0, 0, :, :]
        Y = frames[0, 1, :, :]
        U = frames[0, 2, :, :]
        V = frames[0, 3, :, :]
                
        res_loss = res_loss_fn(X, Y, U, V, pred_temp)
        
        loss = res_loss + fid_loss
        
        loss.backward()
        optimizer.step()

        stats.update(loss.item())

        # Print only when epoch % 10 == 0
        if epoch % 10 == 0:
            print(f'Epoch [{epoch}], Iteration [{iter_idx + 1}/{len(dataloader)}], Loss: {loss.item():.5f}')
            
    # If epoch is 500, save the last prediction to a .mat file
    if epoch == 500:
        prediction_np = pred_temp.cpu().detach().numpy()
        savemat(f'prediction_epoch_{epoch}.mat', {'prediction': prediction_np})

    return stats.avg



def main():  
    args = get_args()
    print(myutils.gct(), f'Args = {args}')

    #if args.log:
    if not os.path.exists('logs'):
        os.makedirs('logs')

    log_dir = 'logs/{}'.format(time.strftime('%Y%m%d-%H%M'))
    log_path = os.path.join(log_dir, 'log')
    model_path = os.path.join(log_dir, 'model')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    myutils.save_scripts(log_dir, scripts_to_save=glob('*.*'))
    myutils.save_scripts(log_dir, scripts_to_save=glob('dataset/*.py', recursive=True))
    myutils.save_scripts(log_dir, scripts_to_save=glob('model/*.py', recursive=True))
    myutils.save_scripts(log_dir, scripts_to_save=glob('myutils/*.py', recursive=True))

    print(myutils.gct(), f'Create log dir: {log_dir}')
    
    # Ensure 'device' is defined outside of main if not already done
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    print(f"Using device: {device}")
    
    dataset = loadmat('data_input.mat')
    
    # Convert to PyTorch tensors and stack
    # Initialize tensors from your dataset with requires_grad=True for X and Y only
    X = torch.tensor(dataset['X'], dtype=torch.float32, requires_grad=False)
    Y = torch.tensor(dataset['Y'], dtype=torch.float32, requires_grad=False)
    U = torch.tensor(dataset['U'], dtype=torch.float32, requires_grad=False)
    V = torch.tensor(dataset['V'], dtype=torch.float32, requires_grad=False)
    
    # Stack all tensors to form the input_frames tensor
    input_frames = torch.stack([X, Y, U, V], dim=0)
    
    target_frame = torch.tensor(dataset['h'], dtype=torch.float32)
    
    # (N, C, H, W) format is expected for images
    # Example assumes data_channels is C x H x W; adjust as necessary
    input_frames = input_frames.unsqueeze(0)  # Add a batch dimension if needed
    
    # Prepare DataLoader
    # TensorDataset and DataLoader expect data to be in (N, C, H, W) format for images
    dataset = TensorDataset(input_frames)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=10, pin_memory=True)
    
    print(myutils.gct(), 'Load training cases.')

    model = AutoEncoder(device)
    model = model.to(device)
    model.train()
    model.apply(myutils.set_bn_eval)  # turn-off BN

    params = model.parameters()
    optimizer = torch.optim.AdamW(filter(lambda x: x.requires_grad, params), args.lr)

    start_epoch = 0
    best_loss = 100000000

    if args.seed < 0:
        seed = int(time.time())
    else:
        seed = args.seed

    print(myutils.gct(), 'Random seed:', seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    fid_loss_fn = torch.nn.SmoothL1Loss().to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.scheduler_step, gamma=0.5, last_epoch=start_epoch - 1)

    for epoch in range(start_epoch, args.total_epochs):

        lr = scheduler.get_last_lr()[0]
        
        if epoch % 10 == 0:
            print('')
            print(myutils.gct(), f'Epoch: {epoch} lr: {lr}')

        loss = train_model(device, epoch, model, dataloader, fid_loss_fn, optimizer, target_frame)
        if args.log:

            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': loss,
                'seed': seed,
            }

            checkpoint_path = f'{model_path}/final.pth'
            torch.save(checkpoint, checkpoint_path)

            if best_loss > loss:
                best_loss = loss

                # don't save model very often for memory efficiency
                # checkpoint_path = f'{model_path}/epoch_{epoch:03d}_loss_{loss:.03f}.pth'
                #torch.save(checkpoint, checkpoint_path)

                checkpoint_path = f'{model_path}/best.pth'
                torch.save(checkpoint, checkpoint_path)

                print('Best model updated.')

            # Check if the file exists and is empty; if so, write the header
            if epoch == 0:
                with open(log_path, 'w') as log_file:
                    log_file.write('Epoch, Loss\n')
                    
            # Create a formatted string for the log values
            log_values = f'{epoch}, {loss.item():.5e}\n'
            # Write the log values to a file
            with open(log_path, 'a') as log_file:
                log_file.write(log_values)

        scheduler.step()


if __name__ == '__main__':
    start_time = time.time()  # Record start time
    print(myutils.gct(), 'Training starts.')
    main()  # Assuming main() is your training function
    end_time = time.time()  # Record end time
    training_time = end_time - start_time  # Calculate training time

    print(myutils.gct(), 'Training done.')
    print("Training time:", training_time, "seconds.")
