import numpy as np
import torch

from utils import *




def get_gaussian_noise(data, dist='G', noise_std =50, mode='S', min_noise = 0, max_noise = 55):
    '''
    code modified from: https://github.com/sreyas-mohan/udvd/blob/main/utils/noise_model.py
    '''
    if(dist == 'G'):
        noise_std /= 255.
        min_noise /= 255.
        max_noise /= 255.
        # print(data.shape)
        noise = np.random.randn(*data.shape)
        # noise = torch.randint(data.shape)
        if mode == 'B':
            n = noise.shape[0];
            noise_tensor_array = (max_noise - min_noise) * torch.rand(n) + min_noise;
            for i in range(n):
                noise.data[i] = noise.data[i] * noise_tensor_array[i];
        else:
            noise = noise * noise_std;
    elif(dist == 'P'):
        # noise = torch.randn_like(data);
        noise = np.random.randn(*data.shape)
        if mode == 'S':
            noise_std /= 255.
            # noise = torch.poisson(data*noise_std)/noise_std - data
            noise = np.random.poisson(data*noise_std)/noise_std - data
    return noise


def add_poisson_noise(image, noise_std):
    # Scale the image values to [0, 1]
    # image = image.astype(np.float32) / 255.0
    noise_std=noise_std/255
    # Generate Poisson noise
    noise = np.random.poisson(image * noise_std) / noise_std

    return noise


def add_impulse_noise(image, noise_level):
    # Generate random noise mask
    
    mask = np.random.random(image.shape)

    # Set pixels to salt (maximum intensity) noise
    image[mask < noise_level/2] = 1.0

    # Set pixels to pepper (minimum intensity) noise
    image[mask > 1 - noise_level/2] = 0.0

    return image

