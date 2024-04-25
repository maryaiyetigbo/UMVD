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



class BatchedSingleVideo(torch.utils.data.Dataset):
    def __init__(self, data_path, clean_path, noise_path, dataset="DAVIS", video="boat", patch_size=None, stride=64, n_frames=5,
                 aug=0, dist="G", mode="S", noise_std=30, min_noise=0, max_noise=100, sample=True, heldout=False):
        super().__init__()
        self.data_path = data_path
        self.noise_path = noise_path
        self.dataset = dataset
        self.size = patch_size
        self.stride = stride
        self.n_frames = n_frames
        self.aug = aug
        self.heldout = heldout

        if dataset == "DAVIS":
            self.files = sorted(glob.glob(os.path.join(data_path, video, clean_path, "*.jpg")))
            self.noisy_files = sorted(glob.glob(os.path.join(data_path, video, noise_path, "*.jpg")))
            # print(os.listdir(os.path.join(data_path, video, clean_path)))
            # print(glob.glob(os.path.join(data_path, video, clean_path, "*.jpg")))
            # print(self.files[0], self.noisy_files[0])
        elif dataset == "gopro_540p" or dataset == "Derfs" or dataset == "Short_Derfs":
            self.files = sorted(glob.glob(os.path.join(data_path, dataset, clean_path, "*.png")))
            self.noisy_files = sorted(glob.glob(os.path.join(data_path, dataset, noise_path, "*.png")))
        elif dataset == "LiveHDR":
            self.files = sorted(glob.glob(os.path.join(data_path, clean_path, "*.jpg")))[:200]
            # print(sorted(glob.glob(os.path.join(data_path, clean_path, "*.jpg"))))[:200]
            # print(os.path.join(data_path, clean_path, "*.jpg"))
            # print(self.files)
            self.noisy_files = sorted(glob.glob(os.path.join(data_path, noise_path, "*.jpg")))[:200]
            # print(os.path.join(data_path, noise_path, "*.jpg"))
            # print(self.noisy_files)
        # elif dataset == "Vid3oC":
        #     self.files = sorted(glob.glob(os.path.join(data_path, "TrainingHR", video, "*.png")))
        # elif dataset == "Nanoparticles":
        #     self.files = sorted(glob.glob(os.path.join(data_path, "*.png")))
        #     self.noisy_files = sorted(glob.glob(os.path.join(data_path, "*.npy")))

        self.len = self.bound = len(self.files)
        if self.heldout:
            self.len -= 5
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.reverse = transforms.Compose([transforms.ToPILImage()])

        Img = Image.open(self.files[0])
        Img = np.array(Img)
        if dataset == "Nanoparticles":
            self.H, self.W = Img.shape
        else:
            self.H, self.W, C = Img.shape

        if self.size is not None:
            # self.n_H = (int((self.H-self.size)/self.stride)+1)
            # self.n_W = (int((self.W-self.size)/self.stride)+1)
            self.n_H = (math.ceil((self.H-self.size)/self.stride)+1)
            self.n_W = (math.ceil((self.W-self.size)/self.stride)+1)
            self.n_patches = self.n_H * self.n_W
            self.len *= self.n_patches

        self.hflip = transforms.Compose([transforms.RandomHorizontalFlip(p=1)])
        self.vflip = transforms.Compose([transforms.RandomVerticalFlip(p=1)])

        if aug >= 1: # Horizonatal and Vertical Flips
            self.len *= 4
        if aug >= 2: # Reverse the Video
            self.len *= 2
        if aug >= 3: # Variable Frame Rate
            self.len *= 4

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        hop = 1
        reverse = 0
        flip = 0
        if self.aug >= 3: # Variable Frame Rate
            hop = index % 4 + 1
            index = index // 4
        if self.aug >= 2: # Reverse the Video
            reverse = index % 2
            index = index // 2
        if self.aug >= 1: # Horizonatal and Vertical Flips
            flip = index % 4
            index = index // 4

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
        
        # if self.dataset == "Nanoparticles":
        #     N, H, W = Img.shape
        # else:
        #     N, H, W, C = Img.shape
        # if self.dataset == "Nanoparticles":
        #     Img = Img.reshape(H, W, 1)
        
        noisy_Img = Image.open(self.noisy_files[index])
        noisy_Img = np.array(noisy_Img)
        if self.dataset=="gopro_540p" or self.dataset == "Derfs" or self.dataset == "Short_Derfs":
            noisy_Img = noisy_Img[...,:3]
        noisy_Img = np.expand_dims(noisy_Img, axis=0)
        

        for i in range(hop, x+1, hop):
            end = max(0, ends)
            off = max(0,i-x+end)
            img = Image.open(self.files[index-i+off])
            img = np.array(img)
            img = np.expand_dims(img, axis=0)
            
            # if self.dataset == "Nanoparticles":
            #     img = img.reshape(N, H, W, 1)
            # noisy_img = np.load(self.noisy_files[index-i+off])
            noisy_img = Image.open(self.noisy_files[index-i+off])
            noisy_img = np.array(noisy_img)
            if self.dataset=="gopro_540p" or self.dataset == "Derfs" or self.dataset == "Short_Derfs":
                noisy_img = noisy_img[...,:3]
            noisy_img = np.expand_dims(noisy_img, axis=0)
            
            if reverse == 0:
                Img = np.concatenate((img, Img), axis=0)
                noisy_Img = np.concatenate((noisy_img, noisy_Img), axis=0)
            else:
                Img = np.concatenate((Img, img), axis=0)
                noisy_Img = np.concatenate((noisy_Img, noisy_img), axis=0)
                

        for i in range(hop, x+1, hop):
            end = -min(0,ends)
            off = max(0,i-x+end)
            img = Image.open(self.files[index+i-off])
            img = np.array(img)
            img = np.expand_dims(img, axis=0)
            
            # if self.dataset == "Nanoparticles":
            #     img = img.reshape(N, H, W, 1)
            # noisy_img = np.load(self.noisy_files[index+i-off])
            noisy_img = Image.open(self.noisy_files[index+i-off])
            noisy_img = np.array(noisy_img)
            if self.dataset=="gopro_540p" or self.dataset == "Derfs" or self.dataset == "Short_Derfs":
                noisy_img = noisy_img[...,:3]
            noisy_img = np.expand_dims(noisy_img, axis=0)
            
            if reverse == 0:
                Img = np.concatenate((Img, img), axis=0)
                noisy_Img = np.concatenate((noisy_Img, noisy_img), axis=0)
            else:
                Img = np.concatenate((img, Img), axis=0)
                noisy_Img = np.concatenate((noisy_img, noisy_Img), axis=0)
                
        
        # print(Img.shape)
            

        if self.size is not None:
            # nh = (patch // self.n_W)*self.stride
            # nw = (patch % self.n_W)*self.stride
            
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

        if flip == 1:
            Img = np.stack([np.flip(Img[i], 1) for i in range(self.n_frames)], axis=0)
            noisy_Img = np.stack([np.flip(noisy_Img[i], 1) for i in range(self.n_frames)], axis=0)
        elif flip == 2:
            Img = np.stack([np.flip(Img[i], 0) for i in range(self.n_frames)], axis=0)
            noisy_Img = np.stack([np.flip(noisy_Img[i], 0) for i in range(self.n_frames)], axis=0)
        elif flip == 3:
            Img = np.stack([np.flip(Img[i], (1,0)) for i in range(self.n_frames)], axis=0)
            noisy_Img = np.stack([np.flip(noisy_Img[i], (1,0)) for i in range(self.n_frames)], axis=0)
            
        # for i in range(self.n_frames):
        #     print(noisy_Img[i].max())
        #     plt.imshow(noisy_Img[i])
        #     plt.show()
        # print(Img.shape)
            
        Img = torch.stack([self.transform(np.array(Img[i])).type(torch.FloatTensor) for i in range(self.n_frames)], axis=0)
        noisy_Img = torch.stack([self.transform(np.array(noisy_Img[i])).type(torch.FloatTensor) for i in range(self.n_frames)], axis=0)
            
        # print(Img.shape, noisy_Img.shape)
        # return torch.tensor(np.array(Img)).type(torch.FloatTensor), torch.tensor(np.array(noisy_Img)).type(torch.FloatTensor)
        return Img, noisy_Img

    




class CTC(torch.utils.data.Dataset):
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
                img = cv2.resize(img, (int(img.shape[1]*0.5), int(img.shape[0]*0.5)), interpolation=cv2.INTER_AREA)
                (h, w) = img.shape
                # (h, w) = np.array(cv2.imread(files[0], cv2.IMREAD_GRAYSCALE)).shape
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
        # files = self.files
        # print(files)
        # print(index)
        img = cv2.imread(files[index], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (int(img.shape[1]*0.5), int(img.shape[0]*0.5)), interpolation=cv2.INTER_AREA)
        (h, w) = np.array(img).shape
        Img = np.reshape(np.array(img), (h,w,1))

        for i in range(1,x+1):
            end = max(0, ends)
            off = max(0,i-x+end)
            img = cv2.imread(files[index-i+off], cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (int(img.shape[1]*0.5), int(img.shape[0]*0.5)), interpolation=cv2.INTER_AREA)
            img = np.reshape(np.array(img), (h,w,1))
            Img = np.concatenate((img, Img), axis=2)

        for i in range(1,x+1):
            end = -min(0,ends)
            off = max(0,i-x+end)
            img = cv2.imread(files[index+i-off], cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (int(img.shape[1]*0.5), int(img.shape[0]*0.5)), interpolation=cv2.INTER_AREA)
            img = np.reshape(np.array(img), (h,w,1))
            Img = np.concatenate((Img, img), axis=2)

        if self.size is not None:
            nh = (patch // nW)*self.stride
            nw = (patch % nW)*self.stride
            Img = Img[nh:(nh+self.size), nw:(nw+self.size), :]

        return self.transform(Img).type(torch.FloatTensor), self.transform(Img).type(torch.FloatTensor)    
    
    
    
    
class CTC_Batched(torch.utils.data.Dataset):
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
                img = cv2.resize(img, (int(img.shape[1]*0.5), int(img.shape[0]*0.5)), interpolation=cv2.INTER_AREA)
                (h, w) = img.shape
                # (h, w) = np.array(cv2.imread(files[0], cv2.IMREAD_GRAYSCALE)).shape
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
        # files = self.files
        # print(files)
        # print(index)
        img = cv2.imread(files[index], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (int(img.shape[1]*0.5), int(img.shape[0]*0.5)), interpolation=cv2.INTER_AREA)
        # (h, w) = np.array(img).shape
        Img = np.expand_dims(np.array(img), axis=0)
        # print(Img.shape)

        for i in range(1,x+1):
            end = max(0, ends)
            off = max(0,i-x+end)
            img = cv2.imread(files[index-i+off], cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (int(img.shape[1]*0.5), int(img.shape[0]*0.5)), interpolation=cv2.INTER_AREA)
            img = np.expand_dims(np.array(img), axis=0)
            Img = np.concatenate((img, Img), axis=0)

        for i in range(1,x+1):
            end = -min(0,ends)
            off = max(0,i-x+end)
            img = cv2.imread(files[index+i-off], cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (int(img.shape[1]*0.5), int(img.shape[0]*0.5)), interpolation=cv2.INTER_AREA)
            img = np.expand_dims(np.array(img), axis=0)
            Img = np.concatenate((Img, img), axis=0)

        if self.size is not None:
            nh = (patch // nW)*self.stride
            nw = (patch % nW)*self.stride
            Img = Img[:, nh:(nh+self.size), nw:(nw+self.size)]
        # print(Img.shape)
        # self.transform(np.array(Img[i]))
            
        Img = torch.stack([self.transform(np.array(Img[i])).type(torch.FloatTensor) for i in range(self.n_frames)], axis=0)

        return Img, Img
    
    

class MUSCData_Batched(torch.utils.data.Dataset):
    def __init__(self, data_path, start_size=0, end_size=1000, patch_size=None, stride=64, n_frames=5):
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
            dataset = hdf.get('1')
            self.files =np.array(dataset[self.start_size:self.end_size])
    
        self.folders = []
        # for folder in parent_folders:
        #     self.folders.append(os.path.join(folder, "01"))
        #     self.folders.append(os.path.join(folder, "02"))
        #for folder in self.folders:
        #files = sorted(glob.glob(os.path.join(data_path, "*.png")))
        if self.size is not None:
            #(h, w) = np.array(cv2.imread(files[0], cv2.IMREAD_GRAYSCALE)).shape
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
        #print(self.bounds)
        # print(self.folders)
        #

        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        ends = 0
        x = (self.n_frames-1) // 2
        #print('dataloader index')
        #print(self.nHs)
        for i, bound in enumerate(self.bounds):
            #print('\nbound-', bound, 'next idx-',index)
            if index < bound:
                #print('i=',i)
                #print('working on this bound: ', bound)
                # print('This index: ',index)
                #folder = self.folders[i-1]
                index -= self.bounds[i-1]
                newbound = bound - self.bounds[i-1]
                # print('folder:', folder)
                # print('new index:', index)
                # print('new bound:', newbound)
                if self.size is not None:
                    nH = self.nHs[i-1]
                    #print(nH)
                    nW = self.nWs[i-1]
                    patch = index % (nH*nW)
                    index = index // (nH*nW)
                    # print(patch)
                    newbound = newbound // (nH*nW)
                if(index < x):
                    ends = x-index
                elif(newbound-1-index < x):
                    ends = -(x-(newbound-1-index))
                break

        
        # with h5py.File(self.data_path, 'r') as hdf:
        #     dataset = hdf.get('1')
        #     files =np.array(dataset[self.start_size:self.end_size])
            
        #files = sorted(glob.glob(os.path.join(self.data_path, "*.png")))
        #img = cv2.imread(files[index], cv2.IMREAD_GRAYSCALE)
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
        # print(Img.shape)
        # self.transform(np.array(Img[i]))
            
        Img = torch.stack([self.transform(np.array(Img[i])).type(torch.FloatTensor) for i in range(self.n_frames)], axis=0)

        return Img, Img
    
    
    
    


    
    
class SingleVideo(torch.utils.data.Dataset):
    def __init__(self, data_path, clean_path, noise_path, dataset="DAVIS", video="boat", patch_size=None, stride=64, n_frames=5,
                 aug=0, dist="G", mode="S", noise_std=30, min_noise=0, max_noise=100, sample=True, heldout=False):
        super().__init__()
        self.data_path = data_path
        self.noise_path = noise_path
        self.dataset = dataset
        self.size = patch_size
        self.stride = stride
        self.n_frames = n_frames
        self.aug = aug
        self.heldout = heldout

        if dataset == "DAVIS":
            self.files = sorted(glob.glob(os.path.join(data_path, video, clean_path, "*.jpg")))
            self.noisy_files = sorted(glob.glob(os.path.join(data_path, video, noise_path, "*.jpg")))
            # print(os.listdir(os.path.join(data_path, video, noise_path)))
            # print(glob.glob(os.path.join(data_path, video, clean_path, "*.jpg")))
        elif dataset == "gopro_540p" or dataset == "Derfs" or dataset == "Short_Derfs":
            self.files = sorted(glob.glob(os.path.join(data_path, dataset, clean_path, "*.png")))
            self.noisy_files = sorted(glob.glob(os.path.join(data_path, dataset, noise_path, "*.png")))
        elif dataset == "LiveHDR":
            self.files = sorted(glob.glob(os.path.join(data_path, clean_path, "*.jpg")))[:200]
            self.noisy_files = sorted(glob.glob(os.path.join(data_path, noise_path, "*.jpg")))[:200]

        # print(len(self.noisy_files))
        self.len = self.bound = len(self.files)
        if self.heldout:
            self.len -= 5
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.reverse = transforms.Compose([transforms.ToPILImage()])

        Img = Image.open(self.files[0])
        Img = np.array(Img)
        # if dataset == "Nanoparticles":
        if dataset == "Nanoparticles":
            self.H, self.W = Img.shape
        else:
            self.H, self.W, C = Img.shape


        if self.size is not None:
            # self.n_H = (int((H-self.size)/self.stride)+1)
            # self.n_W = (int((W-self.size)/self.stride)+1)
            
            self.n_H = (math.ceil((self.H-self.size)/self.stride)+1)
            self.n_W = (math.ceil((self.W-self.size)/self.stride)+1)
            self.n_patches = self.n_H * self.n_W
            self.len *= self.n_patches

        self.hflip = transforms.Compose([transforms.RandomHorizontalFlip(p=1)])
        self.vflip = transforms.Compose([transforms.RandomVerticalFlip(p=1)])

        if aug >= 1: # Horizonatal and Vertical Flips
            self.len *= 4
        if aug >= 2: # Reverse the Video
            self.len *= 2
        if aug >= 3: # Variable Frame Rate
            self.len *= 4

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        
        hop = 1
        reverse = 0
        flip = 0
        if self.aug >= 3: # Variable Frame Rate
            hop = index % 4 + 1
            index = index // 4
        if self.aug >= 2: # Reverse the Video
            reverse = index % 2
            index = index // 2
        if self.aug >= 1: # Horizonatal and Vertical Flips
            flip = index % 4
            index = index // 4

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
        if self.dataset == "Nanoparticles":
            H, W = Img.shape
        else:
            H, W, C = Img.shape
        if self.dataset == "Nanoparticles":
            Img = Img.reshape(H, W, 1)
            
        # print(index)
        
        # noisy_Img = np.load(self.noisy_files[index])
        noisy_Img = Image.open(self.noisy_files[index])
        noisy_Img = np.array(noisy_Img)
        if self.dataset=="gopro_540p" or self.dataset == "Derfs":
            noisy_Img = noisy_Img[...,:3]

        for i in range(hop, x+1, hop):
            end = max(0, ends)
            off = max(0,i-x+end)
            img = Image.open(self.files[index-i+off])
            img = np.array(img)
            if self.dataset == "Nanoparticles":
                img = img.reshape(H, W, 1)
            # noisy_img = np.load(self.noisy_files[index-i+off])
            noisy_img = Image.open(self.noisy_files[index-i+off])
            noisy_img = np.array(noisy_img)
            if self.dataset=="gopro_540p" or self.dataset == "Derfs":
                noisy_img = noisy_img[...,:3]
                
            if reverse == 0:
                Img = np.concatenate((img, Img), axis=2)
                # noisy_Img = np.concatenate((noisy_img, noisy_Img), axis=0)
                noisy_Img = np.concatenate((noisy_img, noisy_Img), axis=2)
            else:
                Img = np.concatenate((Img, img), axis=2)
                # noisy_Img = np.concatenate((noisy_Img, noisy_img), axis=0)
                noisy_Img = np.concatenate((noisy_Img, noisy_img), axis=2)

        for i in range(hop, x+1, hop):
            end = -min(0,ends)
            off = max(0,i-x+end)
            img = Image.open(self.files[index+i-off])
            img = np.array(img)
            if self.dataset == "Nanoparticles":
                img = img.reshape(H, W, 1)
            # noisy_img = np.load(self.noisy_files[index+i-off])
            noisy_img = Image.open(self.noisy_files[index+i-off])
            noisy_img = np.array(noisy_img)
            if self.dataset=="gopro_540p" or self.dataset == "Derfs":
                noisy_img = noisy_img[...,:3]
                
            if reverse == 0:
                Img = np.concatenate((Img, img), axis=2)
                # noisy_Img = np.concatenate((noisy_Img, noisy_img), axis=0)
                noisy_Img = np.concatenate((noisy_Img, noisy_img), axis=2)
            else:
                Img = np.concatenate((img, Img), axis=2)
                # noisy_Img = np.concatenate((noisy_img, noisy_Img), axis=0)
                noisy_Img = np.concatenate((noisy_img, noisy_Img), axis=2)
            

        if self.size is not None:
            # nh = (patch // self.n_W)*self.stride
            # nw = (patch % self.n_W)*self.stride
            
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
                        
            Img = Img[nh:(nh+self.size), nw:(nw+self.size), :]
            # noisy_Img = noisy_Img[:, nh:(nh+self.size), nw:(nw+self.size)]
            noisy_Img = noisy_Img[nh:(nh+self.size), nw:(nw+self.size), :]

        if flip == 1:
            Img = np.flip(Img, 1)
            # noisy_Img = np.flip(noisy_Img, 2)
            noisy_Img = np.flip(noisy_Img, 1)
        elif flip == 2:
            Img = np.flip(Img, 0)
            # noisy_Img = np.flip(noisy_Img, 1)
            noisy_Img = np.flip(noisy_Img, 0)
        elif flip == 3:
            Img = np.flip(Img, (1,0))
            # noisy_Img = np.flip(noisy_Img, (2,1))
            noisy_Img = np.flip(noisy_Img, (1,0))
        # print(Img.shape, noisy_Img.shape)
        return self.transform(np.array(Img)).type(torch.FloatTensor), self.transform(np.array(noisy_Img)).type(torch.FloatTensor) #,torch.from_numpy(noisy_Img.copy())

    
    
    
    
class MultipleVideo(torch.utils.data.Dataset):
    def __init__(self, data_path, clean_path, noise_path, dataset="DAVIS", video="boat", patch_size=None, stride=64, n_frames=5,
                 aug=0, mode="train", noise_std=30, min_noise=0, max_noise=100, sample=True, heldout=False):
        super().__init__()
        self.data_path = data_path
        self.noise_path = noise_path
        self.dataset = dataset
        self.size = patch_size
        self.stride = stride
        self.n_frames = n_frames
        self.aug = aug
        self.heldout = heldout

        if dataset == "DAVIS":
            # print(os.listdir(os.path.join(data_path)))
            # self.files = sorted(glob.glob(os.path.join(data_path, "JPEGImages", "480p", video, "*.jpg")))
            # self.files = sorted(glob.glob(os.path.join(data_path, clean_path, "*.jpg")))
            # self.noisy_files = sorted(glob.glob(os.path.join(data_path, noise_path, "*.jpg")))
            # print(os.listdir(os.path.join(data_path, video, noise_path)), len(self.files), len(self.noisy_files))
            
            self.files=[]
            for folder in os.listdir(os.path.join(data_path, clean_path)):
                self.files+=(sorted(glob.glob(os.path.join(data_path, clean_path, folder, "*.jpg"))))
                
            self.noisy_files=[]
            for folder in os.listdir(os.path.join(data_path, noise_path)):
                self.noisy_files+=(sorted(glob.glob(os.path.join(data_path, noise_path, folder, "*.jpg"))))
        elif dataset == "GoPro" or dataset == "Derfs":
            self.files = sorted(glob.glob(os.path.join(data_path, video, "*.png")))
        elif dataset == "Vid3oC":
            self.files = sorted(glob.glob(os.path.join(data_path, "TrainingHR", video, "*.png")))
        elif dataset == "Nanoparticles":
            self.files = sorted(glob.glob(os.path.join(data_path, "*.png")))
            self.noisy_files = sorted(glob.glob(os.path.join(data_path, "*.npy")))
            
        if mode=='train':
            self.files=self.files[:-30]
            self.noisy_files=self.noisy_files[:-30]
        if mode=='val':
            self.files=self.files[-30:]
            self.noisy_files=self.noisy_files[-30:]
        else:
            self.files=self.files
            self.noisy_files=self.noisy_files
            

        self.len = self.bound = len(self.files)
        # if self.heldout:
        #     self.len -=30
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.reverse = transforms.Compose([transforms.ToPILImage()])
        

        Img = Image.open(self.files[0])
        Img = np.array(Img)
        if dataset == "Nanoparticles":
            H, W = Img.shape
        else:
            H, W, C = Img.shape

#         if not dataset == "Nanoparticles":
#             os.makedirs(os.path.join(data_path, f"Noisy_Videos_{int(noise_std)}"), exist_ok=True)
#             os.makedirs(os.path.join(data_path, f"Noisy_Videos_{int(noise_std)}", video), exist_ok=True)

#             self.noisy_folder = os.path.join(data_path, f"Noisy_Videos_{int(noise_std)}", video)

#             if sample:
#                 # print('here')
#                 for i in range(self.len):
#                     Img = Image.open(self.files[i])
#                     Img = self.transform(Img)
#                     self.C, self.H, self.W = Img.shape
#                     Noise = utils.get_noise(Img, dist=dist, mode=mode, min_noise=min_noise, max_noise=max_noise, noise_std=noise_std).numpy()
#                     Img = Img + Noise
#                     np.save(os.path.join(self.noisy_folder, os.path.basename(self.files[i])[:-3]+".npy"), Img)
#             self.noisy_files = sorted(glob.glob(os.path.join(self.noisy_folder, "*.npy")))

        if self.size is not None:
            self.n_H = (int((H-self.size)/self.stride)+1)
            self.n_W = (int((W-self.size)/self.stride)+1)
            self.n_patches = self.n_H * self.n_W
            self.len *= self.n_patches

        self.hflip = transforms.Compose([transforms.RandomHorizontalFlip(p=1)])
        self.vflip = transforms.Compose([transforms.RandomVerticalFlip(p=1)])

        if aug >= 1: # Horizonatal and Vertical Flips
            self.len *= 4
        if aug >= 2: # Reverse the Video
            self.len *= 2
        if aug >= 3: # Variable Frame Rate
            self.len *= 4

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        hop = 1
        reverse = 0
        flip = 0
        if self.aug >= 3: # Variable Frame Rate
            hop = index % 4 + 1
            index = index // 4
        if self.aug >= 2: # Reverse the Video
            reverse = index % 2
            index = index // 2
        if self.aug >= 1: # Horizonatal and Vertical Flips
            flip = index % 4
            index = index // 4

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
        if self.dataset == "Nanoparticles":
            H, W = Img.shape
        else:
            H, W, C = Img.shape
        if self.dataset == "Nanoparticles":
            Img = Img.reshape(H, W, 1)
        # noisy_Img = np.load(self.noisy_files[index])
        noisy_Img = Image.open(self.noisy_files[index])
        noisy_Img = np.array(noisy_Img)

        for i in range(hop, x+1, hop):
            end = max(0, ends)
            off = max(0,i-x+end)
            img = Image.open(self.files[index-i+off])
            img = np.array(img)
            if self.dataset == "Nanoparticles":
                img = img.reshape(H, W, 1)
            # noisy_img = np.load(self.noisy_files[index-i+off])
            noisy_img = Image.open(self.noisy_files[index-i+off])
            noisy_img = np.array(noisy_img)
            if reverse == 0:
                Img = np.concatenate((img, Img), axis=2)
                # noisy_Img = np.concatenate((noisy_img, noisy_Img), axis=0)
                noisy_Img = np.concatenate((noisy_img, noisy_Img), axis=2)
            else:
                Img = np.concatenate((Img, img), axis=2)
                # noisy_Img = np.concatenate((noisy_Img, noisy_img), axis=0)
                noisy_Img = np.concatenate((noisy_Img, noisy_img), axis=2)

        for i in range(hop, x+1, hop):
            end = -min(0,ends)
            off = max(0,i-x+end)
            img = Image.open(self.files[index+i-off])
            img = np.array(img)
            if self.dataset == "Nanoparticles":
                img = img.reshape(H, W, 1)
            # noisy_img = np.load(self.noisy_files[index+i-off])
            noisy_img = Image.open(self.noisy_files[index+i-off])
            noisy_img = np.array(noisy_img)
            if reverse == 0:
                Img = np.concatenate((Img, img), axis=2)
                # noisy_Img = np.concatenate((noisy_Img, noisy_img), axis=0)
                noisy_Img = np.concatenate((noisy_Img, noisy_img), axis=2)
            else:
                Img = np.concatenate((img, Img), axis=2)
                # noisy_Img = np.concatenate((noisy_img, noisy_Img), axis=0)
                noisy_Img = np.concatenate((noisy_img, noisy_Img), axis=2)
            

        if self.size is not None:
            nh = (patch // self.n_W)*self.stride
            nw = (patch % self.n_W)*self.stride
            Img = Img[nh:(nh+self.size), nw:(nw+self.size), :]
            # noisy_Img = noisy_Img[:, nh:(nh+self.size), nw:(nw+self.size)]
            noisy_Img = noisy_Img[nh:(nh+self.size), nw:(nw+self.size), :]

        if flip == 1:
            Img = np.flip(Img, 1)
            # noisy_Img = np.flip(noisy_Img, 2)
            noisy_Img = np.flip(noisy_Img, 1)
        elif flip == 2:
            Img = np.flip(Img, 0)
            # noisy_Img = np.flip(noisy_Img, 1)
            noisy_Img = np.flip(noisy_Img, 0)
        elif flip == 3:
            Img = np.flip(Img, (1,0))
            # noisy_Img = np.flip(noisy_Img, (2,1))
            noisy_Img = np.flip(noisy_Img, (1,0))
        # print(Img.shape, noisy_Img.shape)
        return self.transform(np.array(Img)).type(torch.FloatTensor), self.transform(np.array(noisy_Img)).type(torch.FloatTensor) #,torch.from_numpy(noisy_Img.copy())

    
    
    
    
    
class DAVIS(torch.utils.data.Dataset):
    def __init__(self, data_path, datatype="train", noisetype='Gaussian30', patch_size=None, stride=64, n_frames=5):
        super().__init__()
        self.data_path = data_path
        self.datatype = datatype
        self.size = patch_size
        self.stride = stride
        self.n_frames = n_frames
        self.noisetype=noisetype

        if self.datatype == "train":
            self.folders = pd.read_csv(os.path.join(data_path, "ImageSets", "2017", "train.txt"), header=None)
        elif self.datatype == "val":
            self.folders = pd.read_csv(os.path.join(data_path, "ImageSets", "2017", "val.txt"), header=None)
        else:
            self.folders = pd.read_csv(os.path.join(data_path, "ImageSets", "2017", "test-dev.txt"), header=None)
        self.len = 0
        self.bounds = []
        
        # print('folder')

        for folder in self.folders.values:
            files = sorted(glob.glob(os.path.join(data_path, "JPEGImages", "480p", folder[0], "*.jpg")))
            noisy_files = sorted(glob.glob(os.path.join(data_path, "JPEGImages", noisetype, folder[0], "*.jpg")))#the which works with all davis datset
            # files = sorted(glob.glob(os.path.join(data_path, "JPEGImages", "480p", folder[0], "*.png")))
            self.len += len(files)
            self.bounds.append(self.len)

        if self.size is not None:
            self.n_H = (int((480-self.size)/self.stride)+1)
            self.n_W = (int((854-self.size)/self.stride)+1)
            self.n_patches = self.n_H * self.n_W
            self.len *= self.n_patches

        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if self.size is not None:
            patch = index % self.n_patches
            index = index // self.n_patches
            # print(index)

        ends = 0
        x = (self.n_frames-1) // 2
        for i, bound in enumerate(self.bounds):
            if index < bound:
                folder = self.folders.values[i][0]
                if i>0:
                    index -= self.bounds[i-1]
                    newbound = bound - self.bounds[i-1]
                else:
                    newbound = bound
                if(index < x):
                    ends = x-index
                elif(newbound-1-index < x):
                    ends = -(x-(newbound-1-index))
                break

        files = sorted(glob.glob(os.path.join(self.data_path, "JPEGImages", "480p", folder, "*.jpg")))
        noisy_files = sorted(glob.glob(os.path.join(self.data_path, "JPEGImages", self.noisetype, folder, "*.jpg")))

        Img = Image.open(files[index])
        Img = np.array(Img)
        
        NoisyImg = Image.open(noisy_files[index])
        NoisyImg = np.array(NoisyImg)

        for i in range(1,x+1):
            end = max(0, ends)
            off = max(0,i-x+end)
            img = Image.open(files[index-i+off])
            img = np.array(img)
            Img = np.concatenate((img, Img), axis=2)
            
            noisyimg = Image.open(noisy_files[index-i+off])
            noisyimg = np.array(noisyimg)
            NoisyImg = np.concatenate((noisyimg, NoisyImg), axis=2)

        for i in range(1,x+1):
            end = -min(0,ends)
            off = max(0,i-x+end)
            img = Image.open(files[index+i-off])
            img = np.array(img)
            Img = np.concatenate((Img, img), axis=2)
            
            noisyimg = Image.open(noisy_files[index+i-off])
            noisyimg = np.array(noisyimg)
            NoisyImg = np.concatenate((NoisyImg, noisyimg), axis=2)
            

        if self.size is not None:
            nh = (patch // self.n_W)*self.stride
            nw = (patch % self.n_W)*self.stride
            Img = Img[nh:(nh+self.size), nw:(nw+self.size), :]
            NoisyImg = NoisyImg[nh:(nh+self.size), nw:(nw+self.size), :]

        return self.transform(np.array(Img)).type(torch.FloatTensor), self.transform(np.array(NoisyImg)).type(torch.FloatTensor)


    
    
    
class MUSCData(torch.utils.data.Dataset):
    def __init__(self, data_path, start_size=0, end_size=1000, patch_size=None, stride=64, n_frames=5):
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
            dataset = hdf.get('1')
            self.files =np.array(dataset[self.start_size:self.end_size])
    
        self.folders = []
        # for folder in parent_folders:
        #     self.folders.append(os.path.join(folder, "01"))
        #     self.folders.append(os.path.join(folder, "02"))
        #for folder in self.folders:
        #files = sorted(glob.glob(os.path.join(data_path, "*.png")))
        if self.size is not None:
            #(h, w) = np.array(cv2.imread(files[0], cv2.IMREAD_GRAYSCALE)).shape
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
        #print(self.bounds)
        # print(self.folders)
        #

        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        ends = 0
        x = (self.n_frames-1) // 2
        #print('dataloader index')
        #print(self.nHs)
        for i, bound in enumerate(self.bounds):
            #print('\nbound-', bound, 'next idx-',index)
            if index < bound:
                #print('i=',i)
                #print('working on this bound: ', bound)
                # print('This index: ',index)
                #folder = self.folders[i-1]
                index -= self.bounds[i-1]
                newbound = bound - self.bounds[i-1]
                # print('folder:', folder)
                # print('new index:', index)
                # print('new bound:', newbound)
                if self.size is not None:
                    nH = self.nHs[i-1]
                    #print(nH)
                    nW = self.nWs[i-1]
                    patch = index % (nH*nW)
                    index = index // (nH*nW)
                    # print(patch)
                    newbound = newbound // (nH*nW)
                if(index < x):
                    ends = x-index
                elif(newbound-1-index < x):
                    ends = -(x-(newbound-1-index))
                break

        
        # with h5py.File(self.data_path, 'r') as hdf:
        #     dataset = hdf.get('1')
        #     files =np.array(dataset[self.start_size:self.end_size])
            
        #files = sorted(glob.glob(os.path.join(self.data_path, "*.png")))
        #img = cv2.imread(files[index], cv2.IMREAD_GRAYSCALE)
        img = self.files[index]
        (h, w) = np.array(img).shape
        Img = np.reshape(np.array(img), (h,w,1))

        for i in range(1,x+1):
            end = max(0, ends)
            off = max(0,i-x+end)
            #img = cv2.imread(files[index-i+off], cv2.IMREAD_GRAYSCALE)
            img = self.files[index-i+off]
            img = np.reshape(np.array(img), (h,w,1))
            Img = np.concatenate((img, Img), axis=2)

        for i in range(1,x+1):
            end = -min(0,ends)
            off = max(0,i-x+end)
            #img = cv2.imread(files[index+i-off], cv2.IMREAD_GRAYSCALE)
            img = self.files[index+i-off]
            img = np.reshape(np.array(img), (h,w,1))
            Img = np.concatenate((Img, img), axis=2)

        if self.size is not None:
            nh = (patch // nW)*self.stride
            nw = (patch % nW)*self.stride
            Img = Img[nh:(nh+self.size), nw:(nw+self.size), :]
        
        Img = Img.astype(np.float)
        #print(type(Img))
        return self.transform(Img).type(torch.FloatTensor), self.transform(Img).type(torch.FloatTensor)
    
    
    
    
class RawVideo(torch.utils.data.Dataset):
    def __init__(self, data_path, datatype="train", patch_size=None, stride=64, n_frames=5, aug=0,
                 scenes=[7, 8, 9, 10, 11],
                 isos = [1600, 3200, 6400, 12800, 25600]):
        super().__init__()
        self.data_path = data_path
        self.datatype = datatype
        self.size = patch_size
        self.stride = stride
        self.n_frames = n_frames
        self.aug = aug
        
        self.noisy_path = os.path.join(self.data_path, "indoor_raw_noisy")
        self.gt_path = os.path.join(self.data_path, "indoor_raw_gt")
        self.scenes = scenes
        self.isos = isos
        if self.datatype == "train":
            self.nr = 9 # noise_realisations
        elif self.datatype == "val":
            self.nr = 1 # only the 9th noise realisation used for heldout
        elif self.datatype == "test":
            self.nr = 10
        self.fpv = self.bound = 7 # frames_per_video
        
        self.len = self.fpv * self.nr * len(self.isos) * len(self.scenes)
        
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.reverse = transforms.Compose([transforms.ToPILImage()])

        Img = skimage.io.imread(os.path.join(self.noisy_path, f"scene{self.scenes[0]}", 
                                             f"ISO{self.isos[0]}", "frame1_noisy0.tiff"))
        H, W = Img.shape
        
        if self.size is not None:
            self.n_H = (int((H-self.size)/self.stride)+1)
            self.n_W = (int((W-self.size)/self.stride)+1)
            self.n_patches = self.n_H * self.n_W
            self.len *= self.n_patches

        self.hflip = transforms.Compose([transforms.RandomHorizontalFlip(p=1)])
        self.vflip = transforms.Compose([transforms.RandomVerticalFlip(p=1)])

        if aug >= 1: # Horizonatal and Vertical Flips
            self.len *= 4
        if aug >= 2: # Reverse the Video
            self.len *= 2
        
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        hop = 1
        reverse = 0
        flip = 0
        if self.aug >= 2: # Reverse the Video
            reverse = index % 2
            index = index // 2
        if self.aug >= 1: # Horizonatal and Vertical Flips
            flip = index % 4
            index = index // 4

        if self.size is not None:
            patch = index % self.n_patches
            index = index // self.n_patches
            
        scene = index % len(self.scenes)
        index = index // len(self.scenes)
        
        iso = index % len(self.isos)
        index = index // len(self.isos)
        
        if self.datatype == "val":
            nr = 9
        else:
            nr = index % self.nr
            index = index // self.nr
        
        ends = 0
        x = ((self.n_frames-1) // 2)*hop
        if index < x:
            ends = x - index
        elif self.bound-1-index < x:
            ends = -(x-(self.bound-1-index))

        Img = skimage.io.imread(os.path.join(self.gt_path, 
                                             f"scene{self.scenes[scene]}",
                                             f"ISO{self.isos[iso]}", 
                                             f"frame{index+1}_clean_and_slightly_denoised.tiff"))
        H, W = Img.shape
        Img = Img.reshape(H, W, 1)
        # plt.imshow(cv2.cvtColor(Img, cv2.COLOR_BGR2RGB) )
        # plt.show()
        # print(Img.dtype, print(Img.max()))
        # Img = utils.convert_raw_to_srgb(np.array(Img, dtype=np.uint8))
        # plt.imshow(Img)
        # plt.show()
        noisy_Img = skimage.io.imread(os.path.join(self.noisy_path, 
                                             f"scene{self.scenes[scene]}",
                                             f"ISO{self.isos[iso]}", 
                                             f"frame{index+1}_noisy{nr}.tiff"))
        noisy_Img = noisy_Img.reshape(H, W, 1)
        # print('noisy',noisy_Img.dtype)
        # plt.imshow(noisy_Img)
        # plt.show()
        # noisy_Img = utils.convert_raw_to_srgb(noisy_Img)
        # plt.imshow(noisy_Img)
        # plt.show()

        for i in range(hop, x+1, hop):
            end = max(0, ends)
            off = max(0,i-x+end)
            # img = Image.open(self.files[index-i+off])
            img = skimage.io.imread(os.path.join(self.gt_path, 
                                             f"scene{self.scenes[scene]}",
                                             f"ISO{self.isos[iso]}", 
                                             f"frame{index-i+off+1}_clean_and_slightly_denoised.tiff"))
            img = img.reshape(H, W, 1)
            # img = utils.convert_raw_to_srgb(img)
            # print(img.shape)
            # noisy_img = np.load(self.noisy_files[index-i+off])
            noisy_img = skimage.io.imread(os.path.join(self.noisy_path, 
                                             f"scene{self.scenes[scene]}",
                                             f"ISO{self.isos[iso]}", 
                                             f"frame{index-i+off+1}_noisy{nr}.tiff"))
            noisy_img = noisy_img.reshape(H, W, 1)
            # noisy_img = utils.convert_raw_to_srgb(noisy_img)
            
            if reverse == 0:
                Img = np.concatenate((img, Img), axis=2)
                noisy_Img = np.concatenate((noisy_img, noisy_Img), axis=2)
            else:
                Img = np.concatenate((Img, img), axis=2)
                noisy_Img = np.concatenate((noisy_Img, noisy_img), axis=2)

        for i in range(hop, x+1, hop):
            end = -min(0,ends)
            off = max(0,i-x+end)
            # img = Image.open(self.files[index+i-off])
            img = skimage.io.imread(os.path.join(self.gt_path, 
                                             f"scene{self.scenes[scene]}",
                                             f"ISO{self.isos[iso]}", 
                                             f"frame{index+i-off+1}_clean_and_slightly_denoised.tiff"))
            img = img.reshape(H, W, 1)
            # img = utils.convert_raw_to_srgb(img)
            # noisy_img = np.load(self.noisy_files[index+i-off])
            noisy_img = skimage.io.imread(os.path.join(self.noisy_path, 
                                             f"scene{self.scenes[scene]}",
                                             f"ISO{self.isos[iso]}", 
                                             f"frame{index+i-off+1}_noisy{nr}.tiff"))
            noisy_img = noisy_img.reshape(H, W, 1)
            # noisy_img = utils.convert_raw_to_srgb(noisy_img)
            
            if reverse == 0:
                Img = np.concatenate((Img, img), axis=2)
                noisy_Img = np.concatenate((noisy_Img, noisy_img), axis=2)
            else:
                Img = np.concatenate((img, Img), axis=2)
                noisy_Img = np.concatenate((noisy_img, noisy_Img), axis=2)

        if self.size is not None:
            nh = (patch // self.n_W)*self.stride
            nw = (patch % self.n_W)*self.stride
            Img = Img[nh:(nh+self.size), nw:(nw+self.size), :]
            noisy_Img = noisy_Img[nh:(nh+self.size), nw:(nw+self.size), :]

        if flip == 1:
            Img = np.flip(Img, 1)
            noisy_Img = np.flip(noisy_Img, 1)
        elif flip == 2:
            Img = np.flip(Img, 0)
            noisy_Img = np.flip(noisy_Img, 0)
        elif flip == 3:
            Img = np.flip(Img, (1,0))
            noisy_Img = np.flip(noisy_Img, (1,0))
            
        Img = Img.astype(np.float32)
        noisy_Img = noisy_Img.astype(np.float32)
        Img = (Img-240)/(2**12-1-240)
        noisy_Img = (noisy_Img-240)/(2**12-1-240)

        return self.transform(np.array(Img)).type(torch.FloatTensor), self.transform(np.array(noisy_Img)).type(torch.FloatTensor)    
    
    

    
    
class CalciumData(torch.utils.data.Dataset):
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

        parent_folders = sorted([x for x in glob.glob(os.path.join(data_path, "*/*")) if os.path.isdir(x)])
        self.folders = []
        # for folder in parent_folders:
        #     self.folders.append(os.path.join(folder, "01"))
        #     self.folders.append(os.path.join(folder, "02"))
        #for folder in self.folders:
        self.files = sorted(glob.glob(os.path.join(data_path, "*.png")))
        # self.files = os.listdir(data_path)
        # print(self.files)
        # self.len = self.bound = len(self.files)
        if self.size is not None:
            (h, w) = np.array(cv2.imread(self.files[0], cv2.IMREAD_GRAYSCALE)).shape
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
        # print(self.bounds)
        # print(self.folders)
        #

        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        # print(index)
        ends = 0
        x = (self.n_frames-1) // 2
        # print('dataloader index')
        #print(index)
        for i, bound in enumerate(self.bounds):
            #print(i)
           # print('\nbound-', bound, 'next idx-',index)
            if index < bound:
                # print('i=',i)
                # print('working on this bound: ', bound)
                # print('This index: ',index)
                #folder = self.folders[i-1]
                index -= self.bounds[i-1]
                newbound = bound - self.bounds[i-1]
                # print('folder:', folder)
                # print('new index:', index)
                # print('new bound:', newbound)
                if self.size is not None:
                    nH = self.nHs[i-1]
                    #print(nH)
                    nW = self.nWs[i-1]
                    patch = index % (nH*nW)
                    index = index // (nH*nW)
                    # print(patch)
                    newbound = newbound // (nH*nW)
                if(index < x):
                    ends = x-index
                elif(newbound-1-index < x):
                    ends = -(x-(newbound-1-index))
                break

        # files = sorted(glob.glob(os.path.join(self.data_path, "*.png")))
        file=[]
        # if self.size is not None:
        #     patch = index % self.n_patches
        #     index = index // self.n_patches
        # ends = 0
        # x = ((self.n_frames-1) // 2)
        # if index < x:
        #     ends = x - index
        # elif self.bound-1-index < x:
        #     ends = -(x-(self.bound-1-index))

        img = cv2.imread(self.files[index], cv2.IMREAD_GRAYSCALE)
        (h, w) = np.array(img).shape
        Img = np.reshape(np.array(img), (h,w,1))
        # print(" ")
        for i in range(1,x+1):
            end = max(0, ends)
            off = max(0,i-x+end)
            img = cv2.imread(self.files[index-i+off], cv2.IMREAD_GRAYSCALE)
            img = np.reshape(np.array(img), (h,w,1))
            Img = np.concatenate((img, Img), axis=2)
            file.append(index-i+off)
            # print(index-i+off)
 
        file.append(index)
        # print("  ",index)
        for i in range(1,x+1):
            end = -min(0,ends)
            off = max(0,i-x+end)
            img = cv2.imread(self.files[index+i-off], cv2.IMREAD_GRAYSCALE)
            img = np.reshape(np.array(img), (h,w,1))
            Img = np.concatenate((Img, img), axis=2)
            file.append(index+i-off)
            # print(index+i-off)

        if self.size is not None:
            nh = (patch // nW)*self.stride
            nw = (patch % nW)*self.stride
            Img = Img[nh:(nh+self.size), nw:(nw+self.size), :]

        return self.transform(Img).type(torch.FloatTensor), self.transform(Img).type(torch.FloatTensor)#, file
    
    
    






class BatchedVideo(torch.utils.data.Dataset):
    def __init__(self, data_path, clean_path, noise_path, dataset="DAVIS", video="boat", patch_size=None, stride=64, n_frames=5,
                 aug=0, dist="G", mode="S", noise_std=30, min_noise=0, max_noise=100, sample=True, heldout=False):
        super().__init__()
        self.data_path = data_path
        self.noise_path = noise_path
        self.dataset = dataset
        self.size = patch_size
        self.stride = stride
        self.n_frames = n_frames
        self.aug = aug
        self.heldout = heldout

        if dataset == "DAVIS":
            # print(os.listdir(os.path.join(data_path)))
            # self.files = sorted(glob.glob(os.path.join(data_path, "JPEGImages", "480p", video, "*.jpg")))
            self.files = sorted(glob.glob(os.path.join(data_path, video, clean_path, "*.jpg")))
            self.noisy_files = sorted(glob.glob(os.path.join(data_path, video, noise_path, "*.jpg")))
            # print(os.listdir(os.path.join(data_path, video, noise_path)), len(self.files), len(self.noisy_files))
        elif dataset == "gopro_540p" or dataset == "Derfs":
            self.files = sorted(glob.glob(os.path.join(data_path, dataset, clean_path, "*.png")))
            # print(os.path.join(data_path, dataset, clean_path, "*.png"))
            self.noisy_files = sorted(glob.glob(os.path.join(data_path, dataset, noise_path, "*.png")))
            # print(os.path.join(data_path, dataset, noise_path, "*.png"))
            # print(len(self.noisy_files))
        elif dataset == "Vid3oC":
            self.files = sorted(glob.glob(os.path.join(data_path, "TrainingHR", video, "*.png")))
        elif dataset == "Nanoparticles":
            self.files = sorted(glob.glob(os.path.join(data_path, "*.png")))
            self.noisy_files = sorted(glob.glob(os.path.join(data_path, "*.npy")))

        self.len = self.bound = len(self.files)
        if self.heldout:
            self.len -= 5
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.reverse = transforms.Compose([transforms.ToPILImage()])

        Img = Image.open(self.files[0])
        Img = np.array(Img)
        if dataset == "Nanoparticles":
            H, W = Img.shape
        else:
            H, W, C = Img.shape

        if self.size is not None:
            self.n_H = (int((H-self.size)/self.stride)+1)
            self.n_W = (int((W-self.size)/self.stride)+1)
            self.n_patches = self.n_H * self.n_W
            self.len *= self.n_patches

        self.hflip = transforms.Compose([transforms.RandomHorizontalFlip(p=1)])
        self.vflip = transforms.Compose([transforms.RandomVerticalFlip(p=1)])

        if aug >= 1: # Horizonatal and Vertical Flips
            self.len *= 4
        if aug >= 2: # Reverse the Video
            self.len *= 2
        if aug >= 3: # Variable Frame Rate
            self.len *= 4

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        
        if self.size is not None:
            patch = index % self.n_patches
            index = index // self.n_patches

        Img = Image.open(self.files[index])
        Img = np.array(Img)
        
        if self.dataset == "Nanoparticles":
            H, W = Img.shape
        else:
            H, W, C = Img.shape
        if self.dataset == "Nanoparticles":
            Img = Img.reshape(H, W, 1)

        noisy_Img = Image.open(self.noisy_files[index])
        noisy_Img = np.array(noisy_Img)
        
        if self.dataset=="gopro_540p" or self.dataset == "Derfs":
            noisy_Img = noisy_Img[...,:3]
            

        if self.size is not None:
            nh = (patch // self.n_W)*self.stride
            nw = (patch % self.n_W)*self.stride
            Img = Img[nh:(nh+self.size), nw:(nw+self.size), :]
            # noisy_Img = noisy_Img[:, nh:(nh+self.size), nw:(nw+self.size)]
            noisy_Img = noisy_Img[nh:(nh+self.size), nw:(nw+self.size), :]
            
        return self.transform(np.array(Img)).type(torch.FloatTensor), self.transform(np.array(noisy_Img)).type(torch.FloatTensor) #,torch.from_numpy(noisy_Img.copy())

    
    
    
    
    
    
    
class Naomi_Batched(torch.utils.data.Dataset):
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
    
    
    
    
    
class Naomi(torch.utils.data.Dataset):
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

        Img = np.expand_dims(np.array(img), axis=2)
        noisy_Img = np.expand_dims(np.array(noisy_img), axis=2)

        for i in range(1,x+1):
            end = max(0, ends)
            off = max(0,i-x+end)
            img = self.files[index-i+off]
            img = np.expand_dims(np.array(img), axis=2)
            Img = np.concatenate((img, Img), axis=2)
            
            noisy_img = self.noisy_files[index-i+off]
            noisy_img = np.expand_dims(np.array(noisy_img), axis=2)
            noisy_Img = np.concatenate((noisy_img, noisy_Img), axis=2)

        for i in range(1,x+1):
            end = -min(0,ends)
            off = max(0,i-x+end)
            img = self.files[index+i-off]
            img = np.expand_dims(np.array(img), axis=2)
            Img = np.concatenate((Img, img), axis=2)
            
            noisy_img = self.noisy_files[index+i-off]
            noisy_img = np.expand_dims(np.array(noisy_img), axis=2)
            noisy_Img = np.concatenate((noisy_Img, noisy_img), axis=2)

        if self.size is not None:
            nh = (patch // nW)*self.stride
            nw = (patch % nW)*self.stride
            Img = Img[nh:(nh+self.size), nw:(nw+self.size), :]
            noisy_Img = noisy_Img[nh:(nh+self.size), nw:(nw+self.size), :]
        # print(Img.shape)
        # self.transform(np.array(Img[i]))
        

        return self.transform(np.array(Img)).type(torch.FloatTensor), self.transform(np.array(noisy_Img)).type(torch.FloatTensor)







class NaomiImage_Batched(torch.utils.data.Dataset):
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

        self.files = sorted(glob.glob(os.path.join(clean_path, "*.png")))[:length]
        self.noisy_files = sorted(glob.glob(os.path.join(noise_path, "*.png")))[:length]
        # self.files = tiff.imread(clean_path)[:500]
        # self.noisy_files = tiff.imread(noise_path)[:500]
        
#         self.files = self.files.astype(np.float32)
#         self.files = self.files - self.files.mean()
        
#         self.noisy_files = self.noisy_files.astype(np.float32)
#         self.noisy_files = self.noisy_files - self.noisy_files.mean()
        
        if self.size is not None:
            # print(self.files.shape)
            (h, w) = np.array(cv2.imread(self.files[0], cv2.IMREAD_GRAYSCALE)).shape
            # (h, w) = imageio.imread(self.files[0]).astype(np.float32).shape
            # (h, w) = self.files[0].shape
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

        img = cv2.imread(self.files[index], cv2.IMREAD_GRAYSCALE)
        noisy_img = cv2.imread(self.noisy_files[index], cv2.IMREAD_GRAYSCALE)
        # img = imageio.imread(self.files[index]).astype(np.float32)
        # noisy_img = imageio.imread(self.noisy_files[index]).astype(np.float32)
        # img = self.files[index]
        # noisy_img = self.noisy_files[index]

        Img = np.expand_dims(np.array(img), axis=0)
        noisy_Img = np.expand_dims(np.array(noisy_img), axis=0)

        for i in range(1,x+1):
            end = max(0, ends)
            off = max(0,i-x+end)
            # img = self.files[index-i+off]
            img = cv2.imread(self.files[index-i+off], cv2.IMREAD_GRAYSCALE)
            # img = imageio.imread(self.files[index-i+off]).astype(np.float32)
            img = np.expand_dims(np.array(img), axis=0)
            Img = np.concatenate((img, Img), axis=0)
            
            # noisy_img = self.noisy_files[index-i+off]
            noisy_img = cv2.imread(self.files[index-i+off], cv2.IMREAD_GRAYSCALE)
            # noisy_img = imageio.imread(self.files[index-i+off]).astype(np.float32)
            noisy_img = np.expand_dims(np.array(noisy_img), axis=0)
            noisy_Img = np.concatenate((noisy_img, noisy_Img), axis=0)

        for i in range(1,x+1):
            end = -min(0,ends)
            off = max(0,i-x+end)
            # img = self.files[index+i-off]
            img = cv2.imread(self.files[index+i-off], cv2.IMREAD_GRAYSCALE)
            # img = imageio.imread(self.files[index+i-off]).astype(np.float32)
            img = np.expand_dims(np.array(img), axis=0)
            Img = np.concatenate((Img, img), axis=0)
            
            # noisy_img = self.noisy_files[index+i-off]
            noisy_img = cv2.imread(self.files[index+i-off], cv2.IMREAD_GRAYSCALE)
            # noisy_img = imageio.imread(self.files[index+i-off]).astype(np.float32)
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
    
    
    
    
    
    
    
    
class Invivo_Batched(torch.utils.data.Dataset):
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
    
        self.folders = []
        # for folder in parent_folders:
        #     self.folders.append(os.path.join(folder, "01"))
        #     self.folders.append(os.path.join(folder, "02"))
        #for folder in self.folders:
        #files = sorted(glob.glob(os.path.join(data_path, "*.png")))
        if self.size is not None:
            #(h, w) = np.array(cv2.imread(files[0], cv2.IMREAD_GRAYSCALE)).shape
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
        #print(self.bounds)
        # print(self.folders)
        #

        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        ends = 0
        x = (self.n_frames-1) // 2
        #print('dataloader index')
        #print(self.nHs)
        for i, bound in enumerate(self.bounds):
            #print('\nbound-', bound, 'next idx-',index)
            if index < bound:
                #print('i=',i)
                #print('working on this bound: ', bound)
                # print('This index: ',index)
                #folder = self.folders[i-1]
                index -= self.bounds[i-1]
                newbound = bound - self.bounds[i-1]
                # print('folder:', folder)
                # print('new index:', index)
                # print('new bound:', newbound)
                if self.size is not None:
                    nH = self.nHs[i-1]
                    #print(nH)
                    nW = self.nWs[i-1]
                    patch = index % (nH*nW)
                    index = index // (nH*nW)
                    # print(patch)
                    newbound = newbound // (nH*nW)
                if(index < x):
                    ends = x-index
                elif(newbound-1-index < x):
                    ends = -(x-(newbound-1-index))
                break

        
        # with h5py.File(self.data_path, 'r') as hdf:
        #     dataset = hdf.get('1')
        #     files =np.array(dataset[self.start_size:self.end_size])
            
        #files = sorted(glob.glob(os.path.join(self.data_path, "*.png")))
        #img = cv2.imread(files[index], cv2.IMREAD_GRAYSCALE)
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
        # print(Img.shape)
        # self.transform(np.array(Img[i]))
            
        Img = torch.stack([self.transform(np.array(Img[i])).type(torch.FloatTensor) for i in range(self.n_frames)], axis=0)

        return Img, Img
    
    
    
    

    

class CTC_Batched_old(torch.utils.data.Dataset):
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
            # self.folders.append(os.path.join(folder, "01"))
            self.folders.append(os.path.join(folder, "02"))
        # print(parent_folders)
        for folder in self.folders:
            files = sorted(glob.glob(os.path.join(folder, "*.tif")))
            if self.size is not None:
                img = np.array(cv2.imread(files[0], cv2.IMREAD_GRAYSCALE))
                # img = cv2.resize(img, (int(img.shape[1]*0.5), int(img.shape[0]*0.5)), interpolation=cv2.INTER_AREA)
                (h, w) = img.shape
                # (h, w) = np.array(cv2.imread(files[0], cv2.IMREAD_GRAYSCALE)).shape
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
        # files = self.files
        # print(files)
        # print(index)
        img = cv2.imread(files[index], cv2.IMREAD_GRAYSCALE)
        # img = cv2.resize(img, (int(img.shape[1]*0.5), int(img.shape[0]*0.5)), interpolation=cv2.INTER_AREA)
        # (h, w) = np.array(img).shape
        Img = np.expand_dims(np.array(img), axis=0)
        # print(Img.shape)

        for i in range(1,x+1):
            end = max(0, ends)
            off = max(0,i-x+end)
            img = cv2.imread(files[index-i+off], cv2.IMREAD_GRAYSCALE)
            # img = cv2.resize(img, (int(img.shape[1]*0.5), int(img.shape[0]*0.5)), interpolation=cv2.INTER_AREA)
            img = np.expand_dims(np.array(img), axis=0)
            Img = np.concatenate((img, Img), axis=0)

        for i in range(1,x+1):
            end = -min(0,ends)
            off = max(0,i-x+end)
            img = cv2.imread(files[index+i-off], cv2.IMREAD_GRAYSCALE)
            # img = cv2.resize(img, (int(img.shape[1]*0.5), int(img.shape[0]*0.5)), interpolation=cv2.INTER_AREA)
            img = np.expand_dims(np.array(img), axis=0)
            Img = np.concatenate((Img, img), axis=0)

        # print('test',Img.shape)
        if self.size is not None:
            nh = (patch // nW)*self.stride
            nw = (patch % nW)*self.stride
            Img = Img[:, nh:(nh+self.size), nw:(nw+self.size)]
        # print(Img.shape)
        # self.transform(np.array(Img[i]))
            
        Img = torch.stack([self.transform(np.array(Img[i])).type(torch.FloatTensor) for i in range(self.n_frames)], axis=0)
        # print('test transform',Img.shape) 

        return Img, Img
    