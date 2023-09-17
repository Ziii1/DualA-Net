import numpy as np
import torch
import torch.nn.functional as F
from medpy import metric

def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)


def jac(output, gt):
    output = output > 0.5
    gt = gt > 0.5
    # if torch.is_tensor(output):
    #     output = torch.sigmoid(output).data.cpu().numpy()
    # if torch.is_tensor(gt):
    #     gt = gt.data.cpu().numpy()
    inter = torch.sum((output.byte() + gt.byte()) == 2)
    union = torch.sum((output.byte() + gt.byte()) >= 1)
    js = float(inter) / (float(union) + 1e-6)
    return js


def get_sensitivity(output, gt):  #=recall
    output = output > 0.5
    gt = gt > 0.5
    TP = ((output==1).byte() + (gt==1).byte()) == 2
    FN = ((output==0).byte() + (gt==1).byte()) == 2
    SE = float(torch.sum(TP)) / (float(torch.sum(TP+FN)) + 1e-6)
    return SE


# def get_sensitivity(output, gt): # 求敏感度 se=TP/(TP+FN)
#     SE = 0.
#     output = output > 0.5
#     gt = gt > 0.5
#     TP = ((output==1).byte() + (gt==1).byte()) == 2
#     FN = ((output==0).byte() + (gt==1).byte()) == 2
#     #wfy:batch_num>1时，改进
#     if len(output)>1:
#         for i in range(len(output)):
#             SE += float(torch.sum(TP[i])) / (float(torch.sum(TP[i]+FN[i])) + 1e-6)
#     else:
#         SE = float(torch.sum(TP)) / (float(torch.sum(TP+FN)) + 1e-6) #原本只用这一句
#     #SE = float(torch.sum(TP)) / (float(torch.sum(TP + FN)) + 1e-6)  # 原本只用这一句
#     return SE  #返回batch中所有样本的SE和


def get_specificity(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT > threshold
    # TN : True Negative
    # FP : False Positive
    TN = ((SR == 0).byte() + (GT == 0).byte()) == 2
    FP = ((SR == 1).byte() + (GT == 0).byte()) == 2

    SP = float(torch.sum(TN)) / (float(torch.sum(TN + FP)) + 1e-6)

    return SP


def get_precision(output, gt): #precision=PPV
    output = output > 0.5
    gt = gt > 0.5
    TP = ((output==1).byte() + (gt==1).byte()) == 2
    FP = ((output==1).byte() + (gt==0).byte()) == 2
    PC = float(torch.sum(TP)) / (float(torch.sum(TP+FP)) + 1e-6)

    return PC


def get_F1(output, gt):
    se = get_sensitivity(output, gt)
    pc = get_precision(output, gt)
    f1 = 2*se*pc / (se+pc+1e-6)
    return f1