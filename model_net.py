# coding=utf-8
import pdb
from torchvision.models import resnet

import matplotlib
import matplotlib.pyplot as plt
import argparse
import numpy as np
from sklearn.preprocessing import LabelEncoder

from PIL import Image
import cv2
import random
import os
# import neuron
# import surrogate
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from model_net import *
from conv_dnm import *
from torchvision.models.segmentation import deeplabv3_resnet50
# from spikingjelly.clock_driven import neuron, surrogate, functional
# from spikingjelly.activation_based import neuron, layer
# rom spikingjelly.clock_driven import surrogate, functional

# from spikingjelly.activation_based import neuron, surrogate,  functional
# from torchinfo import summary
bn_momentum = 0.1


class ConvBlock0(nn.Module):
    """(convolution => [BN] => ReLU) * 1"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, dilation_rate=(2, 2), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            # nn.ReLU(inplace=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class ConvBlock1(nn.Module):
    """(convolution => [BN] => ReLU) * 1"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # nn.ReLU(nplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class ConvBlock2_ori(nn.Module):
    """(convolution => [BN] => ReLU) * 1"""

    def __init__(self, in_channels, out_channels, mid_channels=None, dilation_rate=2):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=dilation_rate,
                      dilation=(dilation_rate, dilation_rate),
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # nn.ReLU(nplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)
        
class ConvBlock2(nn.Module):
    """DeepLab style atrous convolution block"""

    def __init__(self, in_channels, out_channels, mid_channels=None, dilation_rate=2):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=dilation_rate,
                      dilation=dilation_rate, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)



class ConvBlock(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            # nn.ReLU(inplace=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True)
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


        
class MRNet(nn.Module):
    def __init__(self, device=DEVICE, M=5, flag=1, flag_deeplab = 1):
        super(MRNet, self).__init__()
        self.deeplab_model = self.build_deeplab_model()
        self.conv1 = ConvBlock1(1, 64)
        self.conv2 = ConvBlock2(64, 64, None, 2)  # in out mid dilation
        self.conv3 = ConvBlock2(64, 64, None, 3)
        self.conv4 = ConvBlock2(64, 64, None, 4)
        self.conv5 = ConvBlock2(64, 64, None, 3)
        self.conv6 = ConvBlock2(128, 64, None, 3)
        self.conv7 = ConvBlock2(64, 64, None, 2)
        self.conv8 = ConvBlock2(128, 64, None, 2)
        self.conv9 = ConvBlock1(64, 64)
        self.conv10 = ConvBlock1(128, 64)
        self.conv2d = nn.Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1))
        self.Dnm_conv2d = conv_dnm(64, 1, M)
        self.flag_deeplab = flag_deeplab
        self.flag = flag
    
           
    def build_deeplab_model(self):
        # 加载预训练的DeepLabv3模型作为Backbone
        deeplab_model = deeplabv3_resnet50(pretrained=True)
        # 修改输入通道数
        deeplab_model.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 修改分类头，将输出通道数设置为1
        # pdb.set_trace()
        deeplab_model.classifier[-1] = nn.Conv2d(256, 64, kernel_size=1, stride=1)
        # deeplab_model.aux_classifier[-1] = nn.Conv2d(256, 384, kernel_size=1, stride=1)
        
        return deeplab_model


    
    def forward(self, x):
        input_size = x.size(0)  # batch_size
        data_x = x
        x1 = self.deeplab_model(x) # deeplab
        # pdb.set_trace()
        x1 = x1['out']
        
        x = self.conv1(x)
        temp1 = x
        x = self.conv2(x)
        temp2 = x
        x = self.conv3(x)
        temp3 = x
        x = self.conv4(x)

        x = self.conv5(x)
        x = torch.cat((x, temp3), axis=1)  # 拼接3和7
        x = self.conv6(x)

        x = self.conv7(x)
        x = torch.cat((x, temp2), axis=1)  # 拼接2和8
        x = self.conv8(x)

        x = self.conv9(x)
        x = torch.cat((x, temp1), axis=1)  # 拼接1和9
        x = self.conv10(x)
        # pdb.set_trace()
        if self.flag_deeplab == 1:
            x = x1 
            
        if self.flag == 1:
            x = self.Dnm_conv2d(x)
        else:
            x = self.conv2d(x)  # deeplab
 
        out = torch.add(data_x, x)
        out0 = F.sigmoid(out)
        return out0, x
        # return out0, F.sigmoid(x)



class RRCNet(nn.Module):
    cnt = 0
    def __init__(self, device=DEVICE, m=10, flag=1, flagMRNET=0, flag_deeplab = 0):
        super(RRCNet, self).__init__()  
        # encoder
        self.enco1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU()
        )
        self.enco2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=bn_momentum),
            nn.ReLU()
        )
        self.enco3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU()
        )
        self.enco4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU()
        )
        self.enco5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU()
        )

        print("Build enceder done..")

        # between encoder and decoder
        self.midco = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU()
        )

        self.out7 = nn.Sequential(
            torch.nn.UpsamplingNearest2d(scale_factor=32),
            nn.Conv2d(512, 1, kernel_size=1, padding=0, bias=False),
            nn.Sigmoid()
            # nn.ReLU(nplace=True),
        )
        
        self.out7_dnm = nn.Sequential(
            torch.nn.UpsamplingNearest2d(scale_factor=32),
            conv_dnm(512, 1, 10),
            nn.Sigmoid()
            # nn.ReLU(nplace=True),
        )
        
        
        # decoder

        self.deco1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU()
        )
        self.out6 = nn.Sequential(
            torch.nn.UpsamplingNearest2d(scale_factor=16),
            nn.Conv2d(512, 1, kernel_size=1, padding=0, bias=False),
            nn.Sigmoid()
            # nn.ReLU(nplace=True),
        )

        self.deco2 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU()
        )
        self.out5 = nn.Sequential(
            torch.nn.UpsamplingNearest2d(scale_factor=8),
            nn.Conv2d(256, 1, kernel_size=1, padding=0, bias=False),
            nn.Sigmoid()
            # nn.ReLU(nplace=True),
        )

        self.out5_dnm = nn.Sequential(
            torch.nn.UpsamplingNearest2d(scale_factor=32),
            conv_dnm(256, 1, 10),
            nn.Sigmoid()
            # nn.ReLU(nplace=True),
        )


        self.deco3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=bn_momentum),
            nn.ReLU()
        )
        self.out4 = nn.Sequential(
            torch.nn.UpsamplingNearest2d(scale_factor=4),
            nn.Conv2d(128, 1, kernel_size=1, padding=0, bias=False),
            nn.Sigmoid()
            # nn.ReLU(nplace=True),
        )
        
        self.out4_dnm = nn.Sequential(
            torch.nn.UpsamplingNearest2d(scale_factor=32),
            conv_dnm(128, 1, 10),
            nn.Sigmoid()
            # nn.ReLU(nplace=True),
        )

        self.deco4 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(),
        )
        self.out3 = nn.Sequential(
            torch.nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(64, 1, kernel_size=1, padding=0, bias=False),
            nn.Sigmoid()
            # nn.ReLU(nplace=True),
        )
        
        self.out3_dnm = nn.Sequential(
            torch.nn.UpsamplingNearest2d(scale_factor=32),
            conv_dnm(64, 1, 5),
            nn.Sigmoid()
            # nn.ReLU(nplace=True),
        )
        

        self.deco5 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(),
            # nn.Conv2d(64, 1, kernel_size=1, stride=1),
            # nn.BatchNorm2d(1, momentum=bn_momentum),
            # nn.ReLU(),
        )
        self.conv2d = nn.Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1))
        self.Dnm_conv2d = conv_dnm(64, 1, m)

        self.flag = flag
        self.mrnet = MRNet(device=DEVICE, M=m, flag=flagMRNET, flag_deeplab = flag_deeplab)

    def forward(self, x):
        id = []
        # encoder
        
        x = self.enco1(x)  # 1-2
        conv_2 = x
        x, id1 = F.max_pool2d(x, kernel_size=(3, 3), stride=2, padding=1, return_indices=True)  # 保留最大值的位置
        id.append(id1)
        x = self.enco2(x)  # 2-4
        conv_4 = x
        x, id2 = F.max_pool2d(x, kernel_size=(3, 3), stride=2, padding=1, return_indices=True)
        id.append(id2)
        x = self.enco3(x)  # 5-7
        conv_7 = x

        x, id3 = F.max_pool2d(x, kernel_size=(3, 3), stride=2, padding=1, return_indices=True)
        id.append(id3)
        x = self.enco4(x)  # 8-10
        conv_10 = x
        x, id4 = F.max_pool2d(x, kernel_size=(3, 3), stride=2, padding=1, return_indices=True)
        id.append(id4)
        x = self.enco5(x)  # 11-13
        conv_13 = x
        x, id5 = F.max_pool2d(x, kernel_size=(3, 3), stride=2, padding=1, return_indices=True)
        id.append(id5)

        # between encoder and decoder
        x = self.midco(x)  # 14-16
        Out7 = self.out7(x)

        # decoder
        x = F.max_unpool2d(x, id[4], kernel_size=2)
        concat_1 = torch.cat((x, conv_13), axis=1)  # 拼接16池化后的结果和13

        x = self.deco1(concat_1)  # 17-19
        Out6 = self.out6(x)
        x = F.max_unpool2d(x, id[3], kernel_size=(2, 2))
        concat_2 = torch.cat((x, conv_10), axis=1)  # 拼接19池化后的结果和10

        x = self.deco2(concat_2)  # 20-22
        Out5 = self.out5(x)
        x = F.max_unpool2d(x, id[2], kernel_size=(2, 2))
        concat_3 = torch.cat((x, conv_7), axis=1)  # 拼接22池化后的结果和7

        x = self.deco3(concat_3)  # 23-25
        Out4 = self.out4(x)
        x = F.max_unpool2d(x, id[1], kernel_size=(2, 2))
        concat_4 = torch.cat((x, conv_4), axis=1)  # 拼接25池化后的结果和4

        x = self.deco4(concat_4)  # 26-27
        Out3 = self.out3(x)
        # Out3 = self.out3_dnm(x)
        x = F.max_unpool2d(x, id[0], kernel_size=(2, 2))
        concat_5 = torch.cat((x, conv_2), axis=1)  # 拼接27池化后的结果和2

        x = self.deco5(concat_5)  # 28-29
        # pdb.set_trace()
        if self.flag == 1:
            x = self.Dnm_conv2d(x)
        else:
            x = self.conv2d(x)
                 
        Out1 = x
        
        Out0, out_m = self.mrnet(Out1)
        # Out2 = F.sigmoid(x)    
        # x = F.sigmoid(x)
        
        # pdb.set_trace()
        ''' 
        pillow_feature(torch.where(Out1 > 0.5, 1., 0.), RRCNet.cnt, flag=0)
        pillow_feature(torch.where(out_m > 0.5, 1., 0.), RRCNet.cnt, flag=1)
        pillow_feature(torch.where(Out0 > 0.5, 1., 0.), RRCNet.cnt, flag=2)
        RRCNet.cnt+=1
        '''
        return Out0, F.sigmoid(Out1), Out3, Out4, Out5, Out6, Out7, out_m
        # out2 + out_m - out_f == out1 (2+9-8=1)  ,out_f,out_m

def pillow_feature(x, cnt, flag):
    # 假设 x 是包含特征的 PyTorch 张量
    x = x.cpu().numpy()  # 将特征转换为 NumPy 数组

    # 可视化特征
    for i in range(x.shape[0]):
        if flag == 0:     # 0 2 4 6 8 10 ---DNM1
            plt.imshow(x[i][0], cmap='viridis')  # 假设每个特征是灰度图像
            # pdb.set_trace()
            plt.axis('off')
            # plt.title(f'Batch {cnt}_{i}_DNM1.png')
            plt.savefig(f'./pic_feature/batch_{cnt}_{i}_DNM0.png')
        elif flag == 1:
            plt.imshow(x[i][0], cmap='viridis')  # 假设每个特征是灰度图像
            plt.axis('off')
            # plt.title(f'Batch {cnt}_{i}_DNM0.png')
            plt.savefig(f'./pic_feature/batch_{cnt}_{i}_DNM1.png')
        else:             # 1 3 5 7 9 11 ---DNM2
            plt.imshow(x[i][0], cmap='viridis')  # 使用 'gray_r' colormap，反转颜色
            # plt.imshow(x[i][0], cmap='gray_r')  # 使用 'gray_r' colormap，反转颜色
            plt.axis('off')
            # plt.title(f'Batch {cnt}_{i}_DNM2.png')
            plt.savefig(f'./pic_feature/batch_{cnt}_{i}_DNM2.png')
    # plt.show()
    
    

# model = RRCNet().cuda()
# # print(model)
# #
# summary(model, (3, 384, 384))
# print(model(torch.rand(4, 3, 384, 384).cuda()).shape)
