import numpy as np
import random
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

seed_val=44
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)



def temporalfilter(n_frames, mid, level=1, minv=0, maxv=1):
    mid=n_frames//2
    x = np.linspace(0, 2*mid, n_frames)
    y = np.array([(1-(xi/mid)**level) if xi <= mid else  (1-((2*mid-xi)/mid)**level) for xi in x])
    y = y*(maxv-minv)+ minv
    return y


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()
        self.replicate = nn.ReplicationPad2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, bias=bias)
        # self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.replicate(x)
        x = self.conv(x)
        x = self.relu(x)
        return x
    

class Pool(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.pool(x)
        return x

    
    
class DepthwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, group, bias=False):
        super().__init__()
        self.replicate = nn.ReplicationPad2d(1)
        self.depthconv = nn.Conv2d(in_channels, out_channels, kernel_size=3, groups=group, bias=bias)
        # self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.replicate(x)
        x = self.depthconv(x)
        x = self.relu(x)
        return x

    
class ENC_Conv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, bias=False, reduce=True):
        super().__init__()
        self.reduce = reduce
        self.conv1 = Conv(in_channels, mid_channels, bias=bias)
        self.conv2 = Conv(mid_channels, mid_channels, bias=bias)
        self.conv3 = Conv(mid_channels, mid_channels, bias=bias)
        self.conv4 = Conv(mid_channels, mid_channels, bias=bias)
        self.conv5 = Conv(mid_channels, out_channels, bias=bias)
        if reduce:
            self.pool = Pool()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        if self.reduce:
            x = self.pool(x)
        return x

class DEC_Conv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, bias=False):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = Conv(in_channels, mid_channels, bias=bias)
        self.conv2 = Conv(mid_channels, mid_channels, bias=bias)
        self.conv3 = Conv(mid_channels, mid_channels, bias=bias)
        self.conv4 = Conv(mid_channels, mid_channels, bias=bias)
        self.conv5 = Conv(mid_channels, mid_channels, bias=bias)
        self.conv6 = Conv(mid_channels, out_channels, bias=bias)
        

    def forward(self, x, x_in):
        x = self.upsample(x)

        # Smart Padding
        diffY = x_in.size()[2] - x.size()[2]
        diffX = x_in.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])

        x = torch.cat((x, x_in), dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        
        return x


    
class Denoiser(nn.Module):
    def __init__(self, in_channels=3, n_output=3, filters=21, bias=False, n_frames=7, level=1, minv=0):
        super().__init__()
        self.in_channels = in_channels
        self.bias = bias
        self.n_f = n_frames
        self.c = in_channels
        
        mid_channels = filters*self.c
        self.convout = mid_channels
        group = in_channels
        self.conv1 = DepthwiseConv(in_channels, mid_channels, group=group, bias=bias)
        self.conv2 = DepthwiseConv(mid_channels, mid_channels, group=group, bias=bias)
        self.conv3 = DepthwiseConv(mid_channels, self.convout, group=group, bias=bias)
        
        enc_channels = self.convout*n_frames
        # enc_channels = 3*n_frames
        self.enc1 = ENC_Conv(enc_channels, 48, 48, bias=bias)
        self.enc2 = ENC_Conv(48, 48, 48, bias=bias)
        self.enc3 = ENC_Conv(48, 96, 48, bias=bias, reduce=False)
        self.dec2 = DEC_Conv(96, 96, 96, bias=bias)
        self.dec1 = DEC_Conv(96+enc_channels, 96, 96, bias=bias)
        
        
        self.out1 = nn.Conv2d(96, 384, 1, bias=bias)
        self.out2 = nn.Conv2d(384, 96, 1, bias=bias)
        self.out3 = nn.Conv2d(96, n_output, 1, bias=bias)
        
        mid = n_frames//2
        LEVEL, MINV = level, minv
        self.weights = torch.tensor(temporalfilter(n_frames, mid, LEVEL, MINV)).unsqueeze(1).float().to(device)
        # print(f'LEVEL = {LEVEL}')

    def forward(self, input):
        B, C, H, W = input.shape
        
        x = self.conv1(input)
        x = self.conv2(x)
        x_conv = self.conv3(x)
        

        x = torch.cat([self.weights[i] * x_conv[(i):(i+1),:,:,:] for i in range(self.n_f)], dim=0)
        x = torch.reshape(x, (1, B*self.convout, H, W))
        
        input = x
        
        x1 = self.enc1(input)
        x2 = self.enc2(x1)
        x = self.enc3(x2)
        x = self.dec2(x, x1)
        x = self.dec1(x, input)
        
        x = F.relu_(self.out1(x))
        x = F.relu_(self.out2(x))
        x = self.out3(x)
        
        return x
    
 