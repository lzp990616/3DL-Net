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
from torchmetrics import JaccardIndex
from torchmetrics.functional import dice_score
# from torchmetrics import Dice
# from tensorboardX import SummaryWriter
import dataset
import logging
import model_net
from model_net import *
from dataset import *
from PIL import Image
import pdb
from medpy import metric
from torchvision.datasets import ImageFolder
import os
# from utils.dice_score import multiclass_dice_coeff, dice_coeff
import torchvision.transforms as TF
from torchmetrics.functional import precision_recall
from torchmetrics import Specificity, JaccardIndex
import argparse
from PIL import Image, ImageDraw

parse = argparse.ArgumentParser()
parse = argparse.ArgumentParser()
# parse.add_argument("action", type=str, help="train or test")
parse.add_argument("--batch_size", type=int, default=6)
parse.add_argument("--EPOCH", type=int, default=100)
parse.add_argument("--LR", type=float, default=0.0001)
parse.add_argument("--DEVICE", type=int, default=0)
parse.add_argument("--M", type=int, default=10)
parse.add_argument("--DNM", type=int, default=1)
parse.add_argument("--DNM_MR", type=int, default=1)
parse.add_argument("--flag_deeplab", type=int, default=1)
parse.add_argument("--ckpt", type=str, help="the path of model weight file")
args = parse.parse_args()
if args.DEVICE == 0:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 使用GPU或者CPU训练
else:
    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")  # 使用GPU或者CPU训练

mean_nums = [0.485, 0.456, 0.406]
std_nums = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((384, 384), Image.BILINEAR),
    transforms.ToTensor(),  # Coverts into Tensors
])

transform_test = transforms.Compose([
    transforms.Resize((384, 384), Image.BILINEAR),
    # transforms.Grayscale(num_output_channels=1),
    # transforms.RandomResizedCrop(224),
    # transforms.CenterCrop(224), #Performs Crop at Center and resizes it to 224
    transforms.ToTensor(),
    
    # transforms.Normalize(mean = mean_nums, std=std_nums) # Normalizes
    # transforms.Normalize((.5,.5,.5), (.5,.5,.5))
])

filepath_busi = './data/Dataset_BUSI/Dataset_BUSI_with_GT/'
filepath_bus = './data/BUS/BUS/'
filepath_busi_m = './data/Dataset_BUSI_malignant/Dataset_BUSI_with_GT/'
filepath_cloth = './data/archive/'
filepath_Polyp = './data/Kvasir-SEG/'
filepath_brain = './data/brain_tumor/'

# filepath = filepath_bus
filepath = filepath_bus
imagefilepath = filepath + 'data/train/'
imagefilepath_label = filepath + 'data/trainannot/'

valfilepath = filepath + 'data/val/'
valfilepath_label = filepath + 'data/valannot/'

test_dataset = dataset.Busi(valfilepath, valfilepath_label, transform, transform_test)
# print(train_dataset.class_to_idx)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)


# train_size = len(train_loader.dataset)
# test_size = len(test_loader.dataset) #incorrect
# train_num_batches = len(train_loader)
# test_num_batches = len(test_loader)


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice

GAMMA = 2
class DynamicFocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DynamicFocalLoss, self).__init__()

    def forward(self, inputs, targets, gamma=GAMMA, smooth=1):
        # Ensure inputs and targets have the same shape
        assert inputs.size() == targets.size(), "Inputs and targets must have the same shape"

        # Apply sigmoid activation
        inputs = torch.sigmoid(inputs)

        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Compute binary cross-entropy with smooth term
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        # Compute dynamic ALPHA based on class distribution

        pos_ratio = targets.mean()
        ALPHA = 1 - pos_ratio.clamp(0.25, 0.75)

        # Compute focal loss
        BCE_EXP = torch.exp(-BCE)
        focal_loss = ALPHA * (1 - BCE_EXP) ** gamma * BCE
        # print(pos_ratio)
        # print(ALPHA)
        return focal_loss


# 损失函数和模型调用
# criterion = nn.CrossEntropyLoss()
criterion = nn.BCELoss()
criterion_mse = nn.MSELoss()
criterion_dice = DiceLoss()

model = model_net.RRCNet(m=args.M, flag=args.DNM, flagMRNET=args.DNM_MR, flag_deeplab=args.flag_deeplab)
model.to(DEVICE)

# 预训练模型和优化器的选用：
# pretrained_model = "./model_pth/bus_0.5loss_norm.log.pth"
pretrained_model = "./pic/model/dnm_bus.pth"
# pretrained_model = "./model_pth/bus_RRCnet.pth"
# pretrained_model = "./model_pth/DKnet_bus.log.pth"

# 预训练模型加载
pretrained = 1
if pretrained:
    pretrain_model = model_net.RRCNet(m=args.M, flag=args.DNM, flagMRNET=args.DNM_MR, flag_deeplab=args.flag_deeplab).to(
            DEVICE)
    pre_dic = torch.load(pretrained_model)
    pretrain_model.load_state_dict(pre_dic["model_static_dict"])
    model_dict = model.state_dict()
    pretrained_dict = pretrain_model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}  # 选择相同的部分
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.eval()


def calculate_metric_percase(pred, gt):
    # pdb.set_trace()
    if torch.is_tensor(pred):
        predict = pred.data.cpu().numpy()
    if torch.is_tensor(gt):
        target = gt.data.cpu().numpy()

    pred = numpy.atleast_1d(predict.astype(numpy.bool))
    gt = numpy.atleast_1d(target.astype(numpy.bool))

    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    pre = metric.binary.precision(pred, gt)
    rec = metric.binary.recall(pred, gt)
    spe = metric.binary.specificity(pred, gt)
    return dice, jc, pre, rec, spe


def pillow_feature(x, cnt, flag):
    x = x.cpu().numpy()  # 将特征转换为 NumPy 数组
    # 可视化特征
    for i in range(x.shape[0]):
        if flag == 0:     # 0 2 4 6 8 10 ---DNM1
            plt.imshow(x[i][0], cmap='gray')  # 假设每个特征是灰度图像
            # pdb.set_trace()
            plt.axis('off')
            # plt.title(f'Batch {cnt}_{i}_DNM1.png')
            plt.savefig(f'./pic_feature/batch_{cnt}_{i}_DNM0.png', bbox_inches='tight', pad_inches=0.0)
        elif flag == 1:
            plt.imshow(x[i][0], cmap='gray')  # 假设每个特征是灰度图像
            plt.axis('off')
            # plt.title(f'Batch {cnt}_{i}_DNM0.png')
            plt.savefig(f'./pic_feature/batch_{cnt}_{i}_DNM1.png', bbox_inches='tight', pad_inches=0.0)
        else:             # 1 3 5 7 9 11 ---DNM2
            plt.imshow(x[i][0], cmap='gray')  # 使用 'gray_r' colormap，反转颜色  viridis
            # plt.imshow(x[i][0], cmap='gray_r')  # 使用 'gray_r' colormap，反转颜色
            plt.axis('off')
            # plt.title(f'Batch {cnt}_{i}_DNM2.png')
            plt.savefig(f'./pic_feature/batch_{cnt}_{i}_DNM2.png', bbox_inches='tight', pad_inches=0.0)

def test():
    for batch_idx, (img, mask_true) in tqdm(enumerate(test_loader)):
        img, label = img.to(DEVICE), mask_true.to(DEVICE)
        with torch.no_grad():
            output = model(img)
            # pdb.set_trace()
            # mask_pred = torch.where(output[0] > 0.5, 1., 0.)
            # mask_pred = mask_pred.float().cpu()
            
            '''
            outputm = torch.where((torch.where(output[0] > 0.5, 1., 0.) - torch.where(output[1] > 0.5, 1., 0.))> 0.5, 1., 0.)
            mask_pred = torch.where((output[1]+outputm+output[0]) > 0.5, 1., 0.).float().cpu()
            
            dice, jc, pre, rec, spe= calculate_metric_percase(mask_pred, mask_true)
            # mask_true = torch.where(mask_true > 0.5, 1., 0.)
            #for i in range(args.batch_size):
            outputm = torch.where((torch.where(output[1] > 0.5, 1., 0.) - torch.where(output[0] > 0.5, 1., 0.))> 0.5, 1., 0.)
            pillow_feature(x=output[2], cnt=batch_idx, flag=0)
            pillow_feature(x=output[3], cnt=batch_idx, flag=1)
            pillow_feature(x=output[4], cnt=batch_idx, flag=2)
            '''    

def predicted_ori2(epoch):
    model.eval()
    dice_score = 0
    jaccard_score = 0
    pre_score = 0
    recall_score = 0
    spe_score = 0
    sum_total_loss = 0
    loss_sum = [0 for i in range(8)]
    for batch_idx, (img, mask_true) in tqdm(enumerate(test_loader)):
        img, label = img.to(DEVICE), mask_true.to(DEVICE)
        with torch.no_grad():
            output = model(img)
            # mask_pred = (Out1>0.5).float()
            
            mask_pred = torch.where(output[1] > 0.5, 1., 0.)
            mask_true = torch.where(mask_true > 0.5, 1., 0.)
            print(mask_pred)
            
            for i in range(args.batch_size):
                
                contours_true, hierarchy_true = cv2.findContours(
                    mask_true[i].cpu().squeeze().numpy().astype('uint8'),
                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                img_true = transforms.ToTensor()(mask_true[i].cpu().squeeze().numpy().astype('uint8'))
                img_true = img_true * 255
                img_true = img_true.permute(1, 2, 0).numpy()
                img_true = cv2.cvtColor(img_true, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(img_true, contours_true, -1, (0, 0, 255), 1)
                # img_pil = Image.fromarray(img_true.transpose((1, 2, 0)))
                
                # 将原图从tensor类型转为ndarray类型，并将通道顺序从(C, H, W)转为(H, W, C)
                img_np = img[i].cpu().numpy().transpose((1, 2, 0))
                
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                # 在原图上绘制轮廓线
                img_np = (img_np * 255).astype(np.uint8)
                cv2.drawContours(img_np, contours_true, -1, (0, 0, 255), thickness=2)     
                # pdb.set_trace()
                # img_np += img_true
                # img_np.
                # img_np = (img_np * 255).astype(np.uint8)
                # img_np[img_np>255] = 255
                # img_pil = Image.fromarray(img_np)
                
                # img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
                # 保存图片
                # img_pil.save('./pic/{}_{}_mask.png'.format(batch_idx, i))
                cv2.imwrite('./pic/{}_{}_mask.png'.format(batch_idx, i), img_np)

def test_report():
    model.eval()
    dice_score = 0
    jaccard_score = 0
    pre_score = 0
    recall_score = 0
    spe_score = 0
    sum_total_loss = 0
    loss_sum = [0 for i in range(8)]
    for batch_idx, (img, mask_true) in tqdm(enumerate(test_loader)):
        img, label = img.to(DEVICE), mask_true.to(DEVICE)
        with torch.no_grad():
            output = model(img)
            mask_pred = torch.where(output[1] > 0.5, 1., 0.)
            mask_pred = mask_pred.float().cpu()
            mask_true = mask_true.to(device=DEVICE, dtype=torch.long).float().cpu()
            loss = criterion_mse(mask_true, mask_pred)
            sum_total_loss += loss.data.item()

            dice, jc, pre, rec, spe = calculate_metric_percase(mask_pred, mask_true)
            pre_score += pre
            recall_score += rec
            dice_score += dice
            jaccard_score += jc
            spe_score += spe
            # pdb.set_trace()
            # dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
    print("pre_score: \t{:.4f}".format(pre_score / len(test_loader)))
    print("recall_score: \t{:.4f}".format(recall_score / len(test_loader)))
    print("dice_score: \t{:.4f}".format(dice_score / len(test_loader)))
    print("jaccard_score: \t{:.4f}".format(jaccard_score / len(test_loader)))
    print("spe_score: \t{:.4f}".format(spe_score / len(test_loader)))
    print("test_loss: \t{:.4f}".format(sum_total_loss / len(test_loader)))
    # logger.info("pre_score: \t{:.4f}".format(pre_score / len(test_loader)))
    # logger.info("recall_score: \t{:.4f}".format(recall_score / len(test_loader)))
    # logger.info("dice_score: \t{:.4f}".format(dice_score / len(test_loader)))
    # logger.info("jaccard_score: {:.4f}".format(jaccard_score / len(test_loader)))
    # logger.info("spe_score: \t{:.4f}".format(spe_score / len(test_loader)))
    return pre_score / len(test_loader), recall_score / len(test_loader), dice_score / len(
        test_loader), jaccard_score / len(test_loader), spe_score / len(test_loader), sum_total_loss / len(test_loader)



# best = 0.9
if __name__ == '__main__':
    # predicted_ori2(epoch)
    test()
    # test_report()
    # predicted_ori(epoch)
    # predicted_ori2(epoch)
    
