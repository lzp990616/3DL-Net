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

from torchmetrics.functional import precision_recall
from torchmetrics import Specificity, JaccardIndex
from sklearn.metrics import accuracy_score, f1_score
import warnings

warnings.filterwarnings("ignore")
import argparse

parse = argparse.ArgumentParser()
parse = argparse.ArgumentParser()
# parse.add_argument("action", type=str, help="train or test")
parse.add_argument("--log_name", type=str, default="./pic/bus_3dl.log")
parse.add_argument("--batch_size", type=int, default=8)
parse.add_argument("--EPOCH", type=int, default=140)
parse.add_argument("--LR", type=float, default=0.0001)
parse.add_argument("--DEVICE", type=int, default=0)
parse.add_argument("--M", type=int, default=10)
parse.add_argument("--DNM", type=int, default=1)
parse.add_argument("--DNM_MR", type=int, default=1)
parse.add_argument("--flag_deeplab", type=int, default=1)
parse.add_argument("--data_name", type=str, default="bus")
parse.add_argument("--LOSSK", type=float, default=0.1)
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
    transforms.ToTensor(),
    # transforms.RandomResizedCrop(384),#Resizes all images into same dimension
    # transforms.RandomRotation(10),# Rotates the images upto Max of 10 Degrees
    # transforms.RandomHorizontalFlip(p=0.4),#Performs Horizantal Flip over images
    # transforms.RandomVerticalFlip(p=0.4),
    # transforms.ColorJitter(brightness=0.2),
    # transforms.ColorJitter(contrast=0.2),
    # transforms.ColorJitter(hue=0.1),
    # transforms.RandomRotation(15),
    # transforms.RandomRotation(90, expand=True),
    # transforms.RandomHorizontalFlip(p=1.0),
    # transforms.RandomHorizontalFlip(),
    # Coverts into Tensors
    transforms.Normalize(mean=mean_nums, std=std_nums)  # Normalizes
    # transforms.Normalize((.5,.5,.5), (.5,.5,.5))
    # normalize
])

transform_test = transforms.Compose([
    transforms.Resize((384, 384), Image.BILINEAR),
    # transforms.Grayscale(num_output_channels=1),
    # transforms.RandomResizedCrop(224),
    # transforms.CenterCrop(224), #Performs Crop at Center and resizes it to 224
    # transforms.RandomVerticalFlip(p=0.4),
    # transforms.ColorJitter(brightness=0.2),
    # transforms.ColorJitter(contrast=0.2),
    # transforms.ColorJitter(hue=0.1),
    transforms.ToTensor(),
    # transforms.Normalize(mean = [0.485], std=[0.224]) # Normalizes
    # transforms.Normalize((.5,.5,.5), (.5,.5,.5))
])

filepath_busi = './data/Dataset_BUSI/Dataset_BUSI_with_GT/'
filepath_bus = './data/BUS/BUS/'
filepath_cloth = './data/archive/'
filepath_bus = './data/BUS/BUS/'
filepath_STU = './data/STU-Hospital-master/'
filepath_Polyp = './data/Kvasir-SEG/'
filepath_brain = './data/brain_tumor/'
filepath_Covid = './data/COVID-19/'

if args.data_name == "bus":
    filepath = filepath_bus
elif args.data_name == "polyp":
    filepath = filepath_Polyp
elif args.data_name == "stu":
    filepath = filepath_STU
elif args.data_name == "brain":
    filepath = filepath_brain
elif args.data_name == "covid":
    filepath = filepath_Covid
    
    
# filepath = filepath_bus

imagefilepath = filepath + 'data/train/'
imagefilepath_label = filepath + 'data/trainannot/'

valfilepath = filepath + 'data/val/'
valfilepath_label = filepath + 'data/valannot/'

train_dataset = dataset.Busi(imagefilepath, imagefilepath_label, transform, transform_test)
test_dataset = dataset.Busi(valfilepath, valfilepath_label, transform, transform_test)
# print(train_dataset.class_to_idx)


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,  drop_last=True)


# train_size = len(train_loader.dataset)
# test_size = len(test_loader.dataset) #incorrect
# train_num_batches = len(train_loader)
# test_num_batches = len(test_loader)

def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-5):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)



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


ALPHA = 0.8
GAMMA = 2
class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # first compute binary cross-entropy
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP) ** gamma * BCE

        return focal_loss

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
criterion_bce = nn.BCELoss()
# criterion_focal = FocalLoss()
criterion_focal = DynamicFocalLoss()
# model = model_net.RRCNet(m=args.M, flag=args.DNM).to(DEVICE)
model = model_net.RRCNet(m=args.M, flag=args.DNM, flagMRNET=args.DNM_MR, flag_deeplab=args.flag_deeplab).to(
    DEVICE)

model.to(DEVICE)

# 预训练模型和优化器的选用：
# pretrained_model = "./log_beifen_weight/bus_2dnm_adam1e-4_epoch160_batchSize8.log.pth"
# optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.09)
optimizer = optim.Adam(model.parameters(), lr=args.LR)

# 预训练模型加载
pretrained = 0
if pretrained:
    pretrain_model = model_net.RRCNet(DEVICE)
    pre_dic = torch.load(pretrained_model)
    pretrain_model.load_state_dict(pre_dic["model_static_dict"])
    model_dict = model.state_dict()
    pretrained_dict = pretrain_model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}  # 选择相同的部分
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.eval()


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.CRITICAL}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


# logger = get_logger('./log/log_M10_2dnm_busi.log')
# log_name
logger = get_logger('./pic/log/' + 'dnm' + '_' + args.data_name + '_norm.log')
# logger = get_logger('./log/bus_RRCNet_2dnm_4_adam 1e-4.log')
logger.info('start training!')

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger(__name__)

lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.EPOCH, warmup=True)
# 训练
def train(epoch):
    model.train()  # 模型的训练模式
    sum_total_loss_batch = 0
    sum_total_loss = 0
    loss_sum = [0 for i in range(7)]
    pre_score = 0
    recall_score = 0
    dice_score = 0
    jaccard_score = 0
    acc_score = 0
    f1_score = 0
    spe_score = 0

    criterion_loss = criterion_focal


    for batch_idx, (img, label) in enumerate(train_loader):  # 从dataloader中按照dataset.py的顺序进行数据调用，tqdm为循环时间的可视化
        # pdb.set_trace()
        loss = [0 for i in range(8)]
        total_loss = 0
        img, label = img.to(DEVICE), label.to(DEVICE)  # 使用gpu
        model.zero_grad()  # 清梯度
        output = model(img)  # 把数据img 传入模型 获得结果output
        loss[0] = criterion_bce(F.sigmoid(output[0]), label)
        loss[1] = criterion_bce(F.sigmoid(output[1]), label)
        # criterion_dice
        for i in range(2, 7):
            loss[i] = args.LOSSK * criterion_loss(output[i], label)

        # loss[6] = loss[6]
        for i in range(7):
            loss_sum[i] += loss[i].data.item()
        for i in range(7):
            total_loss += loss[i]
        dice, jc, pre, rec, spe, acc, f1 = calculate_metric_percase(torch.where(output[0] > 0.5, 1., 0.).cpu(),
                                                                    label.cpu())
        # print(dice, jc, pre, rec, spe, )
        pre_score += pre
        recall_score += rec
        dice_score += dice
        jaccard_score += jc
        spe_score += spe
        acc_score += acc
        f1_score += f1
        sum_total_loss += total_loss.data.item()
        sum_total_loss_batch += total_loss.data.item()
        total_loss.backward()  # 反向传播当前梯度
        # pdb.set_trace()
        optimizer.step()  # 进行单次优化更新参数
        lr_scheduler.step()
        if batch_idx % 10 == 0 and batch_idx != 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLoss0: {:.6f}\tLoss1: {:.6f}\tLoss2: {:.6f}\tLoss3: {:.6f}\tLoss4: {:.6f}\tLoss5: {:.6f}\t'.format(
                    epoch, batch_idx * len(img), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), sum_total_loss_batch / 10, loss_sum[0] / 10,
                           loss_sum[1] / 10, loss_sum[2] / 10, loss_sum[3] / 10, loss_sum[4] / 10, loss_sum[5] / 10))
            sum_total_loss_batch = 0
            loss_sum = [0 for i in range(8)]
    # pdb.set_trace()

    print("\nTrain Epoch: {}".format(epoch))
    print("pre_score: \t{:.4f}".format(pre_score / len(train_loader)))
    print("recall_score: \t{:.4f}".format(recall_score / len(train_loader)))
    print("dice_score: \t{:.4f}".format(dice_score / len(train_loader)))
    print("jaccard_score: \t{:.4f}".format(jaccard_score / len(train_loader)))
    print("spe_score: \t{:.4f}".format(spe_score / len(train_loader)))
    print("acc_score: \t{:.4f}".format(acc_score / len(train_loader)))
    print("f1_score: \t{:.4f}".format(f1_score / len(train_loader)))
    print('Train_loss: \t{:.4f}'.format((sum_total_loss * args.batch_size) / (len(train_loader.dataset))))
    return (sum_total_loss * args.batch_size) / len(train_loader.dataset)


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
    acc = accuracy_score(np.ravel(pred.astype(np.int)), np.ravel(gt.astype(np.int)))

    if pre + rec == 0:
        f1 = 0.0
    else:
        f1 = (2 * pre * rec) / (pre + rec)
    return dice, jc, pre, rec, spe, acc, f1


def test(epoch):
    model.eval()
    dice_score = 0
    jaccard_score = 0
    pre_score = 0
    recall_score = 0
    spe_score = 0
    sum_total_loss = 0
    acc_score = 0
    f1_score = 0

    loss_sum = [0 for i in range(8)]
    for batch_idx, (img, mask_true) in tqdm(enumerate(test_loader)):
        img, label = img.to(DEVICE), mask_true.to(DEVICE)
        with torch.no_grad():
            output = model(img)
            # mask_pred = (Out1>0.5).float()
            mask_pred = torch.where(output[0] > 0.5, 1., 0.)
            # pdb.set_trace()
            # 可视化
            # plt.imshow(transforms.ToPILImage()(mask_pred.squeeze()), interpolation="bicubic")
            # transforms.ToPILImage()(output[0].squeeze()).show()  # Alternatively
            # plt.imshow(transforms.ToPILImage()(mask_true.squeeze()), interpolation="bicubic")
            # transforms.ToPILImage()(mask_true.squeeze()).show()  # Alternatively
            # transforms.ToPILImage()(Out0[0].squeeze()).show()
            mask_pred = mask_pred.float().cpu()
            # mask_pred.argmax(dim=1)
            mask_true = mask_true.to(device=DEVICE, dtype=torch.long).float().cpu()
            # loss = criterion(mask_true, mask_pred)

            # loss[0] = criterion_mse(output[0], label)
            # loss[1] = criterion_mse(output[1], label)
            # for i in range(2, 8):
            #   loss[i] = criterion(output[i], label)
            # for i in range(8):
            #    loss_sum[i] += loss[i].data.item()

            loss = criterion_mse(mask_true, mask_pred)
            sum_total_loss += loss.data.item()
            # precision_fun(mask_pred, mask_true)
            # specificity_fun(mask_pred, mask_true)
            # recall_fun(mask_pred, mask_true)
            # dice_fun(mask_pred, mask_true)
            # jadcard_fun(mask_pred, mask_true)
            # pre, recall = precision_recall(mask_pred, mask_true.int(), average='micro')
            # specificity = Specificity(average='micro')
            # spe_score += specificity(mask_pred, mask_true.int())
            # pdb.set_trace()
            dice, jc, pre, rec, spe, acc, f1 = calculate_metric_percase(mask_pred, mask_true)

            pre_score += pre
            recall_score += rec
            dice_score += dice
            jaccard_score += jc
            spe_score += spe
            acc_score += acc
            f1_score += f1
            # pdb.set_trace()
            # dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)

    print("Test Epoch: {}".format(epoch))
    print("pre_score: \t{:.4f}".format(pre_score / len(test_loader)))
    print("recall_score: \t{:.4f}".format(recall_score / len(test_loader)))
    print("dice_score: \t{:.4f}".format(dice_score / len(test_loader)))
    print("jaccard_score: \t{:.4f}".format(jaccard_score / len(test_loader)))
    print("spe_score: \t{:.4f}".format(spe_score / len(test_loader)))
    print("acc_score: \t{:.4f}".format(acc_score / len(test_loader)))
    print("f1_score: \t{:.4f}".format(f1_score / len(test_loader)))
    print("test_loss: \t{:.4f}\n".format(sum_total_loss / len(test_loader)))

    return pre_score / len(test_loader), recall_score / len(test_loader), dice_score / len(
        test_loader), jaccard_score / len(test_loader), spe_score / len(test_loader), sum_total_loss / len(
        test_loader), acc_score / len(
        test_loader), f1_score / len(
        test_loader)


# logger.info('finish training!')

def adjust_learning_rate(optimizer, epoch):  # 学习率自动调整
    if epoch % 60 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.8


# best = 0.9
if __name__ == '__main__':
    min = 1
    for epoch in range(1, args.EPOCH + 1):
        train_loss = train(epoch)  # 调用训练函数
        pre, recall, dice, jaccard, spe, test_loss, acc, f1 = test(epoch)
        logger.info(f'Epoch {epoch}: train_loss={train_loss:.4f}, '
                    f'pre={pre:.4f}, recall={recall:.4f}, dice={dice:.4f}, jaccard={jaccard:.4f},spe={spe:.4f},acc={acc:.4f},f1={f1:.4f},test_loss={test_loss:.4f},')
        if min > test_loss:
            min = test_loss
            checkpoint = {
                "model_static_dict": model.state_dict(),
                "epoch": epoch,
                "optimizer_state_dic": optimizer.state_dict()
            }
            torch.save(checkpoint, "./pic/model/" + 'dnm' + '_' + args.data_name + '_norm.pth')
    # sen, spe = test()
    # if epoch > 40:
    #     # sen, spe = test()  # 调用测试函数
    #     # if (sen + spe) / 2 > best:  # 断点存储模型方便下次训练
    #     #     best = (sen + spe) / 2

    # adjust_learning_rate(optimizer, epoch)  # 调用学习率自动调整函数
