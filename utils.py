import h5py
import numpy as np
from PIL import Image
import PIL
import cv2
from statistics import mean
import matplotlib.pyplot as plt
import os 
import math
from numpy import linalg
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch
import torch.nn as nn
import torchvision
import sys
import logging
import rawpy
import torch
import random
import time
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from moviepy.editor import *
from pytorch_msssim import ms_ssim, ssim
from collections import OrderedDict
from numbers import Number
from tqdm import tqdm
import datetime
# from .meters import AverageMeter, RunningAverageMeter, TimeMeter

seed_val=44
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

def prctile_norm(x, min_prc=0, max_prc=100):
    y = (x-np.percentile(x, min_prc))/(np.percentile(x, max_prc)-np.percentile(x, min_prc)+1e-7)
    y[y > 1] = 1
    y[y < 0] = 0
    return y


def linspace_filter(n_frames):
    linspace = torch.cat([torch.linspace(1, 0, steps=(n_frames+1)//2)[:-1], torch.tensor([0]), torch.linspace(0, 1, steps=(n_frames+1) // 2)[1:]])
    weights = linspace.unsqueeze(1)
    return weights


def read_h5_processed(startnum,endnum):
    with h5py.File(path+'New Files/Processed Data/concat_D1-1140A_PA_2022-01-07-07-38-42_video_trig_processed.h5', 'r') as hdf:
        ls = list(hdf.keys())
        #print('List of dataset in the file:\n', ls)
        dataset = hdf.get('1')
        #print(len(dataset))
        data =np.array(dataset[startnum:endnum])
        data = np.expand_dims(data, axis=3)
    return data


def init_logging(log=True, log_file=None, resume_training=True):
    handlers = [logging.StreamHandler()]
    if log and log_file is not None:
        mode = "a" if os.path.isfile(resume_training) else "w"
        handlers.append(logging.FileHandler(log_file, mode=mode))
    logging.basicConfig(handlers=handlers, format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
    logging.info("COMMAND: %s" % " ".join(sys.argv))
    # logging.info("Arguments: {}".format(vars(args)))
    
    
def loss_function(output, gt, mode="mse", device="cpu"):
    if(mode == "mse"):
        loss = F.mse_loss(output, gt, reduction="sum") / (gt.size(0) * 2)
        
    if(mode =="l1"):
        loss = torch.nn.functional.l1_loss(output, gt, reduction="sum")
    return loss
        
        
        
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


def save_video(PATH, testloader, model, videoname, fps, device, n_frames=5, cpf=3, make_vid=False):  
    plt.figure(figsize=(8,10)) 
    frames=[]
    psnrs=[]
    ssims=[]
    datalen = len(testloader)
    with torch.no_grad():
        model.eval()
        for i, (clean, imgs) in enumerate(testloader):
            m=(n_frames//2)
            noise_input = imgs[m:(m+1)][0].permute(1,2,0)#.cpu().numpy()
            noisy_img, clean = imgs.to(device).float(), clean#.cpu().numpy()  
            output, mask  = model(noisy_img)

            psnrs.append(round(psnr_fn(clean[(m):(m+1), :, :, :], output, normalized=True, raw=False),2))
            ssims.append(round(ssim_fn(clean[(m):(m+1), :, :, :], output, normalized=True, raw=False),3))

            denoised_img = output[0].permute(1,2,0).cpu().detach()

            torchvision.utils.save_image(output, f'{PATH}/{videoname}_{str(i).zfill(5)}.jpg')

            denoised_img = (denoised_img - denoised_img.min()) / (denoised_img.max() - denoised_img.min())*255
            noise_input = (noise_input - noise_input.min()) / (noise_input.max() - noise_input.min())*255
            frame = np.append(noise_input, denoised_img, axis=1)
            frame = cv2.cvtColor(np.uint8(frame), cv2.COLOR_BGR2RGB)
            frames.append(frame)

    if make_vid:
        video_path = f'{PATH}/{videoname}.mp4'
        height, width, layers = frames[0].shape  
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        fps = fps
        video = cv2.VideoWriter(video_path, fourcc, fps, (width, height)) 
        # Appending the images to the video one by one
        for i in range(len(frames)): 
            video.write(frames[i])    
        cv2.destroyAllWindows() 
        video.release()

        #convert mp4 to gif
        clip = (VideoFileClip(video_path).resize(1.0))
        clip.write_gif(f'{PATH}/{videoname}.gif')
        
    return psnrs, ssims


   
    
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.item()
        self.val = val / n
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


class RunningAverageMeter(object):
    def __init__(self, momentum=0.98):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if isinstance(val, torch.Tensor):
            val = val.item()
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


class TimeMeter(object):
    def __init__(self, init=0):
        self.reset(init)

    def reset(self, init=0):
        self.init = init
        self.start = time.time()
        self.n = 0

    def update(self, val=1):
        self.n += val

    @property
    def avg(self):
        return self.n / self.elapsed_time

    @property
    def elapsed_time(self):
        return self.init + (time.time() - self.start)

    


class ProgressBar:
    """
    CODE MODIFIED FROM: https://github.com/sreyas-mohan/udvd/blob/main/utils/progress_bar.py
    """
    def __init__(self, iterable, epoch=None, prefix=None, quiet=False):
        self.epoch = epoch
        self.quiet = quiet
        self.prefix = prefix + ' | ' if prefix is not None else ''
        if epoch is not None:
            self.prefix += f"epoch {epoch:02d}"
        self.iterable = iterable if self.quiet else tqdm(iterable, self.prefix, leave=False)

    def __iter__(self):
        return iter(self.iterable)

    def log(self, stats, verbose=False):
        if not self.quiet:
            self.iterable.set_postfix(self.format_stats(stats, verbose), refresh=True)

    def format_stats(self, stats, verbose=False):
        postfix = OrderedDict(stats)
        for key, value in postfix.items():
            if isinstance(value, Number):
                fmt = "{:.3f}" if value > 0.001 else "{:.1e}"
                postfix[key] = fmt.format(value)
            elif isinstance(value, AverageMeter) or isinstance(value, RunningAverageMeter):
                if verbose:
                    postfix[key] = f"{value.avg:.3f} ({value.val:.3f})"
                else:
                    postfix[key] = f"{value.avg:.3f}"
            elif isinstance(value, TimeMeter):
                postfix[key] = f"{value.elapsed_time:.1f}s"
            elif not isinstance(postfix[key], str):
                postfix[key] = str(value)
        return postfix

    def print(self, stats, verbose=False):
        postfix = " | ".join(key + " " + value.strip() for key, value in self.format_stats(stats, verbose).items())
        return f"{self.prefix + ' | ' if self.epoch is not None else ''}{postfix}"
            
    
    
    
from os import path as osp
initialized_logger = {}
def get_logger(logger_name='decay', log_level=logging.INFO, log_folder=None, sequence=None):
    """
    CODE MODIFIED FROM: https://github.com/jiangyitong/RCD/blob/main/basicsr/utils/logger.py
    """
    if log_folder is not None:
        os.makedirs(log_folder, exist_ok=True)
    log_file = os.path.join(log_folder, f'atrain_{sequence}.log')
    logger = logging.getLogger(logger_name)
    format_str = "[%(asctime)s]:  %(message)s"
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(format_str))
    logger.addHandler(stream_handler)
    logger.propagate = False
    logger.setLevel(log_level)
    # add file handler
    file_handler = logging.FileHandler(log_file, 'a')
    file_handler.setFormatter(logging.Formatter(format_str, datefmt="%m-%d-%Y %H:%M:%S"))
    file_handler.setLevel(log_level)
    logger.addHandler(file_handler)
        
    initialized_logger[logger_name] = True
    return logger



def close_logger_handlers(logger):
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)



def get_filename():
    ct=datetime.datetime.now()
    logname = f"{ct.month:02d}-{ct.day:02d}_{ct.hour:02d}:{ct.minute:02d}"
    return logname



def save_checkpoint(checkpoint_dir, step, model, optimizer=None, scheduler=None, score=None, mode="min", save=True):
    """
    CODE MODIFIED FROM: https://github.com/sreyas-mohan/udvd/blob/main/utils/train_utils.py
    """
    assert mode == "min" or mode == "max"
    last_step = getattr(save_checkpoint, "last_step", -1)
    save_checkpoint.last_step = max(last_step, step)
    step_checkpoints = False
    default_score = float("inf") if mode == "min" else float("-inf")
    best_score = getattr(save_checkpoint, "best_score", default_score)
    if (score < best_score and mode == "min") or (score > best_score and mode == "max"):
        save_checkpoint.best_step = step
        save_checkpoint.best_score = score

    # if not no_save and step % save_interval == 0:
    if save:
        os.makedirs(checkpoint_dir, exist_ok=True)
        model = [model] if model is not None and not isinstance(model, list) else model
        optimizer = [optimizer] if optimizer is not None and not isinstance(optimizer, list) else optimizer
        scheduler = [scheduler] if scheduler is not None and not isinstance(scheduler, list) else scheduler
        state_dict = {
            "step": step,
            "score": score,
            "last_step": save_checkpoint.last_step,
            "best_step": save_checkpoint.best_step,
            "best_score": getattr(save_checkpoint, "best_score", None),
            "model": [m.state_dict() for m in model] if model is not None else None,
            "optimizer": [o.state_dict() for o in optimizer] if optimizer is not None else None,
            "scheduler": [s.state_dict() for s in scheduler] if scheduler is not None else None,
            # "args": argparse.Namespace(**{k: v for k, v in vars(args).items() if not callable(v)}),
        }

        if step_checkpoints:
            torch.save(state_dict, os.path.join(checkpoint_dir, "checkpoint{}.pt".format(step)))
        if (score < best_score and mode == "min") or (score > best_score and mode == "max"):
            torch.save(model, os.path.join(checkpoint_dir, "checkpoint_best.pt"))
        if step > last_step:
            torch.save(model, os.path.join(checkpoint_dir, "checkpoint_last.pt"))
