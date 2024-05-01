import argparse
import logging
import sys
import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import cv2
from tqdm.notebook import tqdm
from PIL import Image 
import random
# import imageio as io

import numpy as np
import matplotlib.pyplot as plt
import time

from model import *
from data import *
from utils import *

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# %load_ext autoreload
# %autoreload


parser = argparse.ArgumentParser(allow_abbrev=False)

# Add data arguments
parser.add_argument("--data-path", default="../../../../../../zfs/ailab/Denoising//Dataset/Simulated_2P", help="path to data directory")
parser.add_argument("--log-dir", default='./', type=str, help="path to save training log")
parser.add_argument("--sequence", default='standard', type=str, help="video sequence to denoise")
parser.add_argument("--batch-size", default=None, type=int, help="train batch size")
parser.add_argument("--image-size", default=128, type=int, help="Patch size")
parser.add_argument("--n-frames", default=5, type=int, help="number of input frames")
parser.add_argument("--stride", default=64, type=int, help="stride for patch extraction")
# Add model arguments
# parser.add_argument("--model", default="blind-video-net-4", help="model architecture")
parser.add_argument("--level", default=64, type=int, help="l value for the linear filter that controls the weight of the neighboring frames")
parser.add_argument("--minv", default=0.0, type=float, help="weight value assigned to the central frame")
parser.add_argument("--filters", default=21, type=int, help="number of filters for the group convolution")
parser.add_argument("--in-channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out-channels", default=1, type=int, help="number of output channels")
parser.add_argument("--bias", default=False, type=bool, help="bias term")
# Add loss function
parser.add_argument("--loss_fn", default="mse", help="loss function")
# Add optimization arguments
parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
parser.add_argument("--num-epochs", default=25, type=int)
parser.add_argument("--valid-interval", default=1, type=int, help="evaluate every N epochs")

# Parse twice as model arguments are not known the first time
# parser = utils.add_logging_arguments(parser)
args, _ = parser.parse_known_args()
# models.MODEL_REGISTRY[args.model].add_args(parser)
args = parser.parse_args()

seed_val=44
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


mid = args.n_frames // 2
global_step = -1
start_epoch = 0

clean_path = f"{args.data_path}/Clean/2P_standardClean.tif"
noise_path = f'{args.data_path}/2P_standardNoisy.tif'

# log_dir = f'{repo}/CVMI2024/Naomi/Ablation/{n_frames}_N_Frames/{sequence}_{total_frames}F'
log_dir = args.log_dir
logger = get_logger(log_folder=log_dir, sequence=args.sequence)
logger.info(f'Checkpoint path: {log_dir}')

def load_data(clean_path, noise_path, batch_size, image_size, stride, n_frames):
    train_dataset = Naomi_data(clean_path, noise_path, patch_size=image_size, stride=stride, n_frames=n_frames)
    valid_dataset = Naomi_data(clean_path, noise_path, patch_size=None, stride=stride, n_frames=n_frames)
    print(len(train_dataset), len(valid_dataset))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=2, shuffle=False)
    return train_loader, valid_loader

train_loader, valid_loader = load_data(clean_path=clean_path, noise_path=noise_path,
                                       batch_size=args.batch_size, image_size=args.image_size, stride=args.stride, n_frames=args.n_frames)

# logger.info(f"Batch size:{args.batch_size}, filters:{args.filters}")

model =  Denoiser(in_channels=args.in_channels, n_output=args.out_channels, filters=args.filters, 
                  bias=args.bias, n_frames=args.n_frames, level=args.level, minv=args.minv).to(device)
cpf = model.c # channels per frame

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000], gamma=0.5)
logger.info(f"Built a model consisting of {sum(p.numel() for p in model.parameters()):,} parameters")

# Track moving average of loss values
train_meters = {name: RunningAverageMeter(0.98) for name in (["train_loss", "train_psnr", "train_ssim"])}
valid_meters = {name: AverageMeter() for name in (["valid_loss", "valid_psnr", "valid_ssim"])}


val_loss = np.inf
counter = 0 
patience = 5

for epoch in range(start_epoch, args.num_epochs):
    if epoch %10 == 0:
        optimizer.param_groups[0]["lr"] /= 2
        print('learning rate reduced by factor of 2')
    train_bar = ProgressBar(train_loader, epoch)
    for meter in train_meters.values():
        meter.reset()

    for batch_id, (inputs, noisy_inputs) in enumerate(train_bar):
        model.train()
        global_step += 1
        
        noisy_frame = noisy_inputs[(mid):(mid+1), :, :, :].to(device)
        inputs = inputs.to(device)
        noisy_inputs = noisy_inputs.to(device)

        outputs = model(noisy_inputs)

        loss = loss_function(outputs, noisy_frame, mode=args.loss_fn, device=device)

        model.zero_grad()
        loss.backward()
        optimizer.step()

        train_psnr = psnr_fn(inputs[(mid):(mid+1), :, :, :], outputs)
        train_ssim = ssim_fn(inputs[(mid):(mid+1), :, :, :], outputs)
        train_meters["train_loss"].update(loss.item())
        train_meters["train_psnr"].update(train_psnr.item())
        train_meters["train_ssim"].update(train_ssim.item())

        train_bar.log(dict(**train_meters, lr=optimizer.param_groups[0]["lr"]), verbose=True)


    scheduler.step()
    logger.info(train_bar.print(dict(**train_meters, lr=optimizer.param_groups[0]["lr"]), verbose=False))


####Validation#########
    if (epoch+1) % args.valid_interval == 0:
        model.eval()
        for meter in valid_meters.values():
            meter.reset()

        valid_bar = ProgressBar(valid_loader)
        running_valid_psnr = 0.0
        for sample_id, (sample, noisy_inputs) in enumerate(valid_bar):
            with torch.no_grad():
                val_noisy_frame = noisy_inputs[(mid):(mid+1), :, :, :].to(device)

                sample = sample.to(device)
                noisy_inputs = noisy_inputs.to(device)
                
                val_outputs = model(noisy_inputs)

                loss = loss_function(outputs, noisy_frame, mode=args.loss_fn, device=device)

                valid_psnr = psnr_fn(sample[(mid):(mid+1), :, :, :], val_outputs)
                valid_ssim = ssim_fn(sample[(mid):(mid+1), :, :, :], val_outputs)

                # valid_SNR = calculate_snr(sample[(mid):(mid+1), :, :, :], val_outputs)

                running_valid_psnr += valid_psnr
                valid_meters["valid_loss"].update(loss.item())
                valid_meters["valid_psnr"].update(valid_psnr.item())
                valid_meters["valid_ssim"].update(valid_ssim.item())


                noise_PSNR = psnr_fn(sample[(mid):(mid+1), :, :, :], val_noisy_frame)

                output_img = val_outputs[0].permute(1,2,0).cpu().squeeze().numpy()
                output_img = (output_img - output_img.min()) / (output_img.max() - output_img.min())
                save_folder = f"{log_dir}/Epoch{str(epoch).zfill(2)}"
                # os.makedirs(save_folder, exist_ok=True)
                # plt.imsave(f"{save_folder}/{str(sample_id).zfill(4)}.png", output_img, cmap='gray')

        running_valid_psnr /= (sample_id+1)

        
        if valid_meters["valid_loss"].avg < val_loss:
            val_loss = valid_meters["valid_loss"].avg 
            counter =0 
        else:
            counter += 1

        logger.info(f"EVAL_count{counter}:"+train_bar.print(dict(**valid_meters, lr=optimizer.param_groups[0]["lr"])))

        with open(f'{log_dir}/0snr_results.txt', 'a') as f:
            f.write(f"Epoch{epoch}: {valid_meters['valid_ssim'].avg}\n")

        save_checkpoint(log_dir, epoch+1, model, optimizer, score=valid_meters["valid_loss"].avg, mode="min")


    if counter >= patience:
        # torch.save(model, log_dir+'/checkpoint_last.pt')
        break
close_logger_handlers(logger)


