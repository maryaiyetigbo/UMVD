import os
import os.path
import cv2
import glob
import h5py
from PIL import Image
import skimage
import skimage.io
import random
import math
import tifffile as tiff
import imageio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF

import utils

seed_val=44
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


### CODE MODIFIED FROM: https://github.com/sreyas-mohan/udvd/blob/main/data.py



class BatchedSingleVideo(torch.utils.data.Dataset):
    def __init__(self, data_path, clean_path, noise_path, dataset="LiveHDR", video="1Runner", patch_size=None, stride=64, n_frames=5, heldout=False):
        super().__init__()
        self.data_path = data_path
        self.noise_path = noise_path
        self.dataset = dataset
        self.size = patch_size
        self.stride = stride
        self.n_frames = n_frames
        self.heldout = heldout

        self.files = sorted(glob.glob(os.path.join(data_path, clean_path, "*.jpg")))
        self.noisy_files = sorted(glob.glob(os.path.join(data_path, noise_path, "*.jpg")))

        self.len = self.bound = len(self.files)
        
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.reverse = transforms.Compose([transforms.ToPILImage()])

        Img = Image.open(self.files[0])
        Img = np.array(Img)
        
        self.H, self.W, C = Img.shape

        if self.size is not None:
            self.n_H = (math.ceil((self.H-self.size)/self.stride)+1)
            self.n_W = (math.ceil((self.W-self.size)/self.stride)+1)
            self.n_patches = self.n_H * self.n_W
            self.len *= self.n_patches


    def __len__(self):
        return self.len

    def __getitem__(self, index):
        
        if self.size is not None:
            patch = index % self.n_patches
            index = index // self.n_patches

        ends = 0
        x = ((self.n_frames-1) // 2)*hop
        if index < x:
            ends = x - index
        elif self.bound-1-index < x:
            ends = -(x-(self.bound-1-index))

        Img = Image.open(self.files[index])
        Img = np.array(Img)
        Img = np.expand_dims(Img, axis=0)

        
        noisy_Img = Image.open(self.noisy_files[index]).convert('RGB')
        noisy_Img = np.array(noisy_Img)
        noisy_Img = np.expand_dims(noisy_Img, axis=0)
        

        for i in range(1, x+1, hop):
            end = max(0, ends)
            off = max(0,i-x+end)
            img = Image.open(self.files[index-i+off])
            img = np.array(img)
            img = np.expand_dims(img, axis=0)

            noisy_img = Image.open(self.noisy_files[index-i+off]).convert('RGB')
            noisy_img = np.array(noisy_img)
            noisy_img = np.expand_dims(noisy_img, axis=0)

            Img = np.concatenate((img, Img), axis=0)
            noisy_Img = np.concatenate((noisy_img, noisy_Img), axis=0)
                

        for i in range(1, x+1, hop):
            end = -min(0,ends)
            off = max(0,i-x+end)
            img = Image.open(self.files[index+i-off])
            img = np.array(img)
            img = np.expand_dims(img, axis=0)
            
            noisy_img = Image.open(self.noisy_files[index+i-off]).convert('RGB')
            noisy_img = np.array(noisy_img)
            noisy_img = np.expand_dims(noisy_img, axis=0)
  
            Img = np.concatenate((Img, img), axis=0)
            noisy_Img = np.concatenate((noisy_Img, noisy_img), axis=0)


        if self.size is not None:
            
            nh = (patch // self.n_W)*self.stride
            nw = (patch % self.n_W)*self.stride

            if (nh+self.size)>self.H:
                nh = self.H-self.size
                if nh < 0:
                    nh=0
            if (nw+self.size)>self.W:
                nw = self.W-self.size
                if nw < 0:
                    nw=0
                        
            Img = Img[:, nh:(nh+self.size), nw:(nw+self.size), :]
            noisy_Img = noisy_Img[:, nh:(nh+self.size), nw:(nw+self.size), :]

            
        Img = torch.stack([self.transform(np.array(Img[i])).type(torch.FloatTensor) for i in range(self.n_frames)], axis=0)
        noisy_Img = torch.stack([self.transform(np.array(noisy_Img[i])).type(torch.FloatTensor) for i in range(self.n_frames)], axis=0)
            
        return Img, noisy_Img

    
    
    
    
class Naomi_data(torch.utils.data.Dataset):
    def __init__(self, clean_path, noise_path, patch_size=None, stride=64, n_frames=5, length=500):
        super().__init__()
        # self.data_path = data_path
        self.size = patch_size
        self.stride = stride
        self.len = 0
        self.bounds = [0]
        self.nHs = []
        self.nWs = []
        self.n_frames = n_frames

        # self.files = sorted(glob.glob(os.path.join(data_path, "*.png")))
        self.files = tiff.imread(clean_path)[:length]
        self.noisy_files = tiff.imread(noise_path)[:length]
        
#         self.files = self.files.astype(np.float32)
#         self.files = self.files - self.files.mean()
        
#         self.noisy_files = self.noisy_files.astype(np.float32)
#         self.noisy_files = self.noisy_files - self.noisy_files.mean()
        
        if self.size is not None:
            # print(self.files.shape)
            # (h, w) = np.array(cv2.imread(self.files[0], cv2.IMREAD_GRAYSCALE)).shape
            (h, w) = self.files[0].shape
            self.nH = (int((h-self.size)/self.stride)+1)
            self.nW = (int((w-self.size)/self.stride)+1)
            #print(nH)
            self.n_patches = self.nH * self.nW
            self.len += len(self.files) * self.nH * self.nW
            self.nHs.append(self.nH)
            self.nWs.append(self.nW)
            # print(self.nHs)
        else:
            self.len += len(self.files)
        self.bounds.append(self.len)

        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        # print(index)
        ends = 0
        x = (self.n_frames-1) // 2
        for i, bound in enumerate(self.bounds):
            if index < bound:
                index -= self.bounds[i-1]
                newbound = bound - self.bounds[i-1]
                if self.size is not None:
                    nH = self.nHs[i-1]
                    nW = self.nWs[i-1]
                    patch = index % (nH*nW)
                    index = index // (nH*nW)
                    newbound = newbound // (nH*nW)
                if(index < x):
                    ends = x-index
                elif(newbound-1-index < x):
                    ends = -(x-(newbound-1-index))
                break

        # img = cv2.imread(self.files[index], cv2.IMREAD_GRAYSCALE)
        img = self.files[index]
        noisy_img = self.noisy_files[index]

        Img = np.expand_dims(np.array(img), axis=0)
        noisy_Img = np.expand_dims(np.array(noisy_img), axis=0)

        for i in range(1,x+1):
            end = max(0, ends)
            off = max(0,i-x+end)
            img = self.files[index-i+off]
            img = np.expand_dims(np.array(img), axis=0)
            Img = np.concatenate((img, Img), axis=0)
            
            noisy_img = self.noisy_files[index-i+off]
            noisy_img = np.expand_dims(np.array(noisy_img), axis=0)
            noisy_Img = np.concatenate((noisy_img, noisy_Img), axis=0)

        for i in range(1,x+1):
            end = -min(0,ends)
            off = max(0,i-x+end)
            img = self.files[index+i-off]
            img = np.expand_dims(np.array(img), axis=0)
            Img = np.concatenate((Img, img), axis=0)
            
            noisy_img = self.noisy_files[index+i-off]
            noisy_img = np.expand_dims(np.array(noisy_img), axis=0)
            noisy_Img = np.concatenate((noisy_Img, noisy_img), axis=0)

        if self.size is not None:
            nh = (patch // nW)*self.stride
            nw = (patch % nW)*self.stride
            Img = Img[:, nh:(nh+self.size), nw:(nw+self.size)]
            noisy_Img = noisy_Img[:, nh:(nh+self.size), nw:(nw+self.size)]
        # print(Img.shape)
        # self.transform(np.array(Img[i]))
        
        Img = np.expand_dims(Img, axis=3)
        noisy_Img = np.expand_dims(noisy_Img, axis=3)
        Img = torch.stack([self.transform(np.array(Img[i])).type(torch.FloatTensor) for i in range(self.n_frames)], axis=0)
        noisy_Img = torch.stack([self.transform(np.array(noisy_Img[i])).type(torch.FloatTensor) for i in range(self.n_frames)], axis=0)

        return Img, noisy_Img

    
    
    
class MUSC(torch.utils.data.Dataset):
    def __init__(self, data_path, start_size=0, end_size=1000, patch_size=None, stride=64, n_frames=5, h5file='denoised_dsets'):
        super().__init__()
        self.data_path = data_path
        self.start_size = start_size
        self.end_size = end_size
        self.size = patch_size
        self.stride = stride
        self.len = 0
        self.bounds = [0]
        self.nHs = []
        self.nWs = []
        self.n_frames = n_frames

        parent_folders = sorted([x for x in glob.glob(os.path.join(data_path, "*/*")) if os.path.isdir(x)])
        with h5py.File(self.data_path, 'r') as hdf:
            dataset = hdf.get(h5file)
            self.files =np.array(dataset[self.start_size:self.end_size])
    
        if self.size is not None:
            (h, w) = self.files[0].shape
            nH = (int((h-self.size)/self.stride)+1)
            nW = (int((w-self.size)/self.stride)+1)
            #print(nH)
            self.len += len(self.files)*nH*nW
            self.nHs.append(nH)
            self.nWs.append(nW)
            # print(self.nHs)
        else:
            self.len += len(self.files)
        self.bounds.append(self.len)

        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        ends = 0
        x = (self.n_frames-1) // 2
        for i, bound in enumerate(self.bounds):
            if index < bound:
                index -= self.bounds[i-1]
                newbound = bound - self.bounds[i-1]
                if self.size is not None:
                    nH = self.nHs[i-1]
                    nW = self.nWs[i-1]
                    patch = index % (nH*nW)
                    index = index // (nH*nW)
                    newbound = newbound // (nH*nW)
                if(index < x):
                    ends = x-index
                elif(newbound-1-index < x):
                    ends = -(x-(newbound-1-index))
                break

 
        img = self.files[index]
        # (h, w) = np.array(img).shape
        Img = np.expand_dims(np.array(img), axis=0)
        # print(Img.shape)

        for i in range(1,x+1):
            end = max(0, ends)
            off = max(0,i-x+end)
            img = self.files[index-i+off]
            img = np.expand_dims(np.array(img), axis=0)
            Img = np.concatenate((img, Img), axis=0)

        for i in range(1,x+1):
            end = -min(0,ends)
            off = max(0,i-x+end)
            img = self.files[index+i-off]
            img = np.expand_dims(np.array(img), axis=0)
            Img = np.concatenate((Img, img), axis=0)

        if self.size is not None:
            nh = (patch // nW)*self.stride
            nw = (patch % nW)*self.stride
            Img = Img[:, nh:(nh+self.size), nw:(nw+self.size)]
            
        Img = torch.stack([self.transform(np.array(Img[i])).type(torch.FloatTensor) for i in range(self.n_frames)], axis=0)

        return Img, Img
    
    
    
    
    
class Microscopy(torch.utils.data.Dataset):
    def __init__(self, data_path, patch_size=None, stride=64, n_frames=5):
        super().__init__()
        self.data_path = data_path
        self.size = patch_size
        self.stride = stride
        self.len = 0
        self.bounds = [0]
        self.nHs = []
        self.nWs = []
        self.n_frames = n_frames

        parent_folders = sorted([x for x in glob.glob(os.path.join(data_path, "*")) if os.path.isdir(x)])
        self.folders = []
        for folder in parent_folders:
            self.folders.append(os.path.join(folder, "01"))
            self.folders.append(os.path.join(folder, "02"))

        for folder in self.folders:
            files = sorted(glob.glob(os.path.join(folder, "*.tif")))
            if self.size is not None:
                img = np.array(cv2.imread(files[0], cv2.IMREAD_GRAYSCALE))
                (h, w) = img.shape
                nH = (int((h-self.size)/self.stride)+1)
                nW = (int((w-self.size)/self.stride)+1)
                self.len += len(files)*nH*nW
                self.nHs.append(nH)
                self.nWs.append(nW)
            else:
                self.len += len(files)
            self.bounds.append(self.len)

        self.transform = transforms.Compose([transforms.ToTensor()])
        self.files = files

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        ends = 0
        x = (self.n_frames-1) // 2
        for i, bound in enumerate(self.bounds):
            if index < bound:
                folder = self.folders[i-1]
                index -= self.bounds[i-1]
                newbound = bound - self.bounds[i-1]
                if self.size is not None:
                    nH = self.nHs[i-1]
                    nW = self.nWs[i-1]
                    patch = index % (nH*nW)
                    index = index // (nH*nW)
                    newbound = newbound // (nH*nW)
                if(index < x):
                    ends = x-index
                elif(newbound-1-index < x):
                    ends = -(x-(newbound-1-index))
                break

        files = sorted(glob.glob(os.path.join(folder, "*.tif")))
        
        img = cv2.imread(files[index], cv2.IMREAD_GRAYSCALE)
        Img = np.expand_dims(np.array(img), axis=0)

        for i in range(1,x+1):
            end = max(0, ends)
            off = max(0,i-x+end)
            img = cv2.imread(files[index-i+off], cv2.IMREAD_GRAYSCALE)
            img = np.expand_dims(np.array(img), axis=0)
            Img = np.concatenate((img, Img), axis=0)

        for i in range(1,x+1):
            end = -min(0,ends)
            off = max(0,i-x+end)
            img = cv2.imread(files[index+i-off], cv2.IMREAD_GRAYSCALE)
            img = np.expand_dims(np.array(img), axis=0)
            Img = np.concatenate((Img, img), axis=0)


        if self.size is not None:
            nh = (patch // nW)*self.stride
            nw = (patch % nW)*self.stride
            Img = Img[:, nh:(nh+self.size), nw:(nw+self.size)]
            
        Img = torch.stack([self.transform(np.array(Img[i])).type(torch.FloatTensor) for i in range(self.n_frames)], axis=0)

        return Img, Img