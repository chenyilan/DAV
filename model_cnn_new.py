# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       main
   Project Name:    beamform_Unet_post-process
   Author :         Hengrong LAN
   Date:            2018/12/27
   Device:          GTX1080Ti
-------------------------------------------------
   Change Activity:
                   2018/12/10:
-------------------------------------------------
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import numpy as np




def conv3x3(in_channels, out_channels, stride=1, 
            padding=1, bias=True, groups=1):    
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)

def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))

def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)



class DualConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DualConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels


        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.01)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.01)

        return x


class FinConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(FinConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels


        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.in_channels)
        self.bn2 = nn.BatchNorm2d(self.in_channels)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.01)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.01)

        return x


# Model 1 modified Unet for beamforming
class Reg_net(nn.Module):


    def __init__(self,  in_channels=3):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        super(Reg_net, self).__init__()
        

        self.in_channels = in_channels
       
        self.layer1 = DualConv(1,64)
        self.layer2 = DualConv(64,128)
        self.layer3 = DualConv(128,64)
        self.layer4 = DualConv(64,16)
        
        
        self.con1 = conv1x1(1,1)
        self.con2 = conv1x1(16,1)

        self.layer5 = FinConv(1,1)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)


    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)


    def forward(self, xk, x_grad):
        
        out = self.layer1(xk)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.con2(out) + xk - self.con1(x_grad)
        out = self.layer5(out)
        #print(out.size())
        
        return out

if __name__ == "__main__":
    """
    testing
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device =  torch.device('cuda:1')


    #x = Variable(torch.FloatTensor(np.random.random((1, 1, 2560, 128))),requires_grad = True).to(device)
    img = Variable(torch.FloatTensor(np.random.random((1, 1, 380, 380))), requires_grad=True).to(device)
    x_khalf = Variable(torch.FloatTensor(np.random.random((1, 1, 380, 380))), requires_grad=True).to(device)
    model = Reg_net(in_channels=1).to(device)
    
    out = model(img,x_khalf)
    print(out.shape)
    #out = F.upsample(out, (128, 128), mode='bilinear')
    loss = torch.mean(out)

    loss.backward()

    print(loss)
