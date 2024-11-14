import random
import torch
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF

def rotate_flow(U, V, h, angle):
    if angle == 90:
        U_rot, V_rot, h_rot = -TF.rotate(V, 90), TF.rotate(U, 90), TF.rotate(h, 90)
    elif angle == 180:
        U_rot, V_rot, h_rot = -TF.rotate(U, 180), -TF.rotate(V, 180), TF.rotate(h, 180)
    elif angle == 270:
        U_rot, V_rot, h_rot = TF.rotate(V, 270), -TF.rotate(U, 270), TF.rotate(h, 270)
    else:
        U_rot, V_rot, h_rot = U, V, h  # No rotation if angle is 0
    return U_rot, V_rot, h_rot

def add_noise(U, V, noise_std=0.01):
    noise_U = torch.randn(U.size(), device=U.device) * noise_std
    noise_V = torch.randn(V.size(), device=V.device) * noise_std
    U_noisy, V_noisy = U + noise_U, V + noise_V
    return U_noisy, V_noisy # no noise for measured h

def random_crop_and_resize(U, V, h, crop_size, output_size):
    # Ensure crop size is not larger than the input dimensions
    h, w = U.shape[1:3]
    crop_height = min(crop_size, h)
    crop_width = min(crop_size, w)
    # Generate random crop parameters with adjusted crop size
    i, j, h_crop, w_crop = transforms.RandomCrop.get_params(U, output_size=(crop_height, crop_width))
    # Apply cropping
    U_cropped = TF.crop(U, i, j, h_crop, w_crop)
    V_cropped = TF.crop(V, i, j, h_crop, w_crop)
    h_cropped = TF.crop(h, i, j, h_crop, w_crop)
    return U_cropped, V_cropped, h_cropped

def flip_flow(U, V, h, horizontal=True):
    if horizontal:
        U_flipped = TF.hflip(U) * -1  # Reverse sign for horizontal flip
        V_flipped = TF.hflip(V)
        h_flipped = TF.hflip(h)
    else:
        U_flipped = TF.vflip(U)
        V_flipped = TF.vflip(V) * -1  # Reverse sign for vertical flip
        h_flipped = TF.vflip(h)
    return U_flipped, V_flipped, h_flipped

def translate_flow(U, V, h, max_shift=10):
    shift_x = random.randint(-max_shift, max_shift)
    shift_y = random.randint(-max_shift, max_shift)
    
    U_translated = F.pad(U, (max_shift, max_shift, max_shift, max_shift), mode='constant', value=0)
    V_translated = F.pad(V, (max_shift, max_shift, max_shift, max_shift), mode='constant', value=0)
    h_translated = F.pad(h, (max_shift, max_shift, max_shift, max_shift), mode='constant', value=0)

    U_translated = U_translated[:, max_shift + shift_y : max_shift + shift_y + U.shape[1], 
                                max_shift + shift_x : max_shift + shift_x + U.shape[2]]
    
    V_translated = V_translated[:, max_shift + shift_y : max_shift + shift_y + V.shape[1], 
                                max_shift + shift_x : max_shift + shift_x + V.shape[2]]
    
    h_translated = h_translated[:, max_shift + shift_y : max_shift + shift_y + h.shape[1], 
                                max_shift + shift_x : max_shift + shift_x + h.shape[2]]
    
    return U_translated, V_translated, h_translated

def augment_flow(U, V, h, output_size):
    angle = random.choice([0, 90, 180, 270])
    U, V, h = rotate_flow(U, V, h, angle=angle)
    
    if random.random() < 0.5:
        U, V = add_noise(U, V, noise_std=0.01)

    crop_size = random.randint(int(0.8 * output_size[0]), output_size[0])
    U, V, h = random_crop_and_resize(U, V, h, crop_size=crop_size, output_size=output_size)

    if random.random() < 0.5:
        U, V, h = flip_flow(U, V, h, horizontal=True)
    if random.random() < 0.5:
        U, V, h = flip_flow(U, V, h, horizontal=False)

    if random.random() < 0.5:
        U, V, h = translate_flow(U, V, h, max_shift=10)

    return U, V, h
