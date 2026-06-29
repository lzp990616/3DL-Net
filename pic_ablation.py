import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
# from torchmetrics import Dice
# from tensorboardX import SummaryWriter
import dataset
import logging
from dataset import *
from PIL import Image
import pdb
from medpy import metric
from torchvision.datasets import ImageFolder
import os
# from utils.dice_score import multiclass_dice_coeff, dice_coeff
import torchvision.transforms as TF
import argparse



image_files = [os.path.join("./pic/bus/pic_feature_bus", filename)
               for filename in os.listdir("./pic/bus/pic_feature_bus")
               if filename.endswith(".png")]
            

model_order = {"mask": 0, "DSNet": 1, "DMNet": 2, "Ours": 3}

def sort_key(file_name):
    parts = os.path.basename(file_name).split('_')
    model_name = parts[-1].split('.')[0]
    # pdb.set_trace()
    num_parts = tuple(map(int, parts[:-1]))
    model_order_value = model_order[model_name]
    
    return (num_parts, model_order_value)

image_files = sorted(image_files, key=sort_key)

# pdb.set_trace()


# pdb.set_trace()
# 将图像文件划分为每七个图像的列表
image_lists = [image_files[i:i+4] for i in range(0, len(image_files), 4)]
# pdb.set_trace()
# 遍历每个子列表，并将七个图像放在同一行中
for image_list in image_lists:
    fig, axs = plt.subplots(1, 4, figsize=(5, 3))
    plt.subplots_adjust(wspace=0.04)
    for i, image_file in enumerate(image_list):
        image = plt.imread(image_file)
        axs[i].imshow(image, extent=[0, image.shape[1], 0, image.shape[0]], cmap=None, interpolation='nearest')
        axs[i].axis('off')
        model_name = os.path.basename(image_file).split('_')[2].split('.')[0]  # 从文件名中获取模型名称	
        if model_name == "mask":
            model_name = "Ground Truth"
        elif model_name == "Ours":
            model_name = "3DL-Net"
        axs[i].text(0.5, -0.2, model_name, size=10, ha="center", transform=axs[i].transAxes)  # 在图像下方添加模型名称
    plt.savefig('./pic_row/row_{}.png'.format(image_files.index(image_list[0])), dpi=300, bbox_inches='tight')
    plt.close()

