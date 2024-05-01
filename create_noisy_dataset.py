import os
import numpy as np
from PIL import Image, ImageOps
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

from utils import *


# data_path=f"../../../../../../zfs/Denoising/Dataset/LIVEHFR/"
data_path=f"../../../../../../zfs/ailab/Denoising/Dataset/LIVE_YT_HFR_120fps/"
cleanpath=data_path+'clean/'
videos = os.listdir(cleanpath)



################## GAUSSIAN NOISE #####################
sigmas=[30,50,90]
for sigma in sigmas:                     
    print(f'Gaussian{sigma}')
    for video in videos:
        noisepath = f'{data_path}/Gaussian{str(sigma)}/{video}'
        os.makedirs(noisepath, exist_ok=True)
        files = sorted(os.listdir(f'{cleanpath}{video}'))
        # print(len(files))
        for i in range(len(files)):
            img = np.array(Image.open(f'{cleanpath}{video}/{files[i]}'))/255
            noise = get_gaussian_noise(img, noise_std=sigma, mode='S')
            img = img+noise
            plt.imsave(noisepath+'/'+files[i], np.clip((img),0,1))
                

                
################## POISSON NOISE #####################
lambdas=[30,50,90]
for lmbda in lambdas:              
    print(f'Poisson{lmbda}')
    for video in videos:
        noisepath = f'{data_path}/Poisson{str(lmbda)}/{video}'
        os.makedirs(noisepath, exist_ok=True)
        files = sorted(os.listdir(f'{cleanpath}{video}'))
        # print(len(files))
        for i in range(len(files)):
            img = np.array(Image.open(f'{cleanpath}{video}/{files[i]}'))/255
            noise = add_poisson_noise(img, lmbda)
            img = img+noise
            # img = (img - img.min()) / (img.max() - img.min())*255
            plt.imsave(noisepath+'/'+files[i], np.clip((img),0,1))                
     
    
    
################## IMPULSE NOISE #####################    
alphas=[0.2,0.3,0.4]
for alpha in alphas:         
    print(f'Impulse{alpha}')
    for video in videos:
        noisepath = f'{data_path}/Impulse{str(alpha)}/{video}'
        os.makedirs(noisepath, exist_ok=True)
        files = sorted(os.listdir(f'{cleanpath}{video}'))
        # print(len(files))
        for i in range(len(files)):
            img = np.array(Image.open(f'{cleanpath}{video}/{files[i]}'))/255
            noisy_img = add_impulse_noise(img.copy(), alpha)
            # img = (img - img.min()) / (img.max() - img.min())*255
            plt.imsave(noisepath+'/'+files[i], np.clip((noisy_img),0,1))




