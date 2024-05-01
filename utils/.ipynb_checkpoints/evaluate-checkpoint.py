import torch
import torchvision
import sys
import cv2
import random
from statistics import mean
from moviepy.editor import *


def evaluate(PATH, testloader, model, videoname, fps, device, n_frames=5, cpf=3, make_vid=False):  
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