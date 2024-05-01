import numpy as np
import torch
from numpy import linalg
from pytorch_msssim import ms_ssim, ssim
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def prctile_norm(x, min_prc=0, max_prc=100):
    y = (x-np.percentile(x, min_prc))/(np.percentile(x, max_prc)-np.percentile(x, min_prc)+1e-7)
    y[y > 1] = 1
    y[y < 0] = 0
    return y


def ssim_fn(clean, noisy):
    clean = clean.mul(255).clamp(0, 255) #(B, 1, H, W)
    noisy = noisy.mul(255).clamp(0, 255) #(B, 1, H, W)

    clean = clean.cpu().detach().numpy().astype(np.float32).transpose(0,2,3,1)
    noisy = noisy.cpu().detach().numpy().astype(np.float32).transpose(0,2,3,1)
    
    return np.array([structural_similarity(c, n, channel_axis=-1, data_range=255, multichannel=True) 
                     for c, n in zip(clean, noisy)]).mean()


def psnr_fn(clean, noisy):
    clean = clean.mul(255).clamp(0, 255) #(B, 1, H, W)
    noisy = noisy.mul(255).clamp(0, 255) #(B, 1, H, W)

    clean = clean.cpu().detach().numpy().astype(np.float32).transpose(0,2,3,1)
    noisy = noisy.cpu().detach().numpy().astype(np.float32).transpose(0,2,3,1)
    
    return np.array([peak_signal_noise_ratio(c, n, data_range=255) for c, n in zip(clean, noisy)]).mean()
    
    
    
def SNR(gt, x):
    gt = prctile_norm(torch.squeeze(gt.detach().cpu()))
    x = prctile_norm(torch.squeeze(x.detach().cpu()))
    a = linalg.norm(gt.detach().cpu())**2
    b = (linalg.norm(x.detach().cpu() - gt.detach().cpu())**2)+0.00000001
    snr = 10*np.log10(a/b)
    return snr


def calculate_snr(Sy, Sx):
    """
    Calculate the Signal-to-Noise Ratio (SNR) between two images.
    
    Parameters:
    Sy (torch.Tensor): The tensor representing the signal (the original or reference image).
    Sx (torch.Tensor): The tensor representing the noisy signal (the distorted or noisy image).
    
    Returns:
    float: The SNR value in decibels (dB).
    """
    # Ensure the input tensors are of type float for accurate calculations
    Sy = Sy.type(torch.float32)
    Sx = Sx.type(torch.float32)

    # Calculate the energy of the signal and the noise
    signal_energy = torch.norm(Sy, p=2)**2
    noise_energy = torch.norm(Sx - Sy, p=2)**2

    # Avoid division by zero in case the noise energy is 0
    if noise_energy == 0:
        return torch.tensor(float('inf'))
    
    # Calculate the SNR
    snr = 20 * torch.log10(signal_energy / noise_energy)
    return snr