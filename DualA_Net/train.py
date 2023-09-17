import argparse
import os
from collections import OrderedDict
from glob import glob

# import cv2
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
import albumentations as albu

import archs
import losses
from dataset import Dataset
from metrics import iou_score, dice_coef, jac, get_precision, get_sensitivity
from utils import AverageMeter, str2bool

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  #CPU


ARCH_NAMES = archs.__all__  # 'UNet', 'NestedUNet','U_Net','AttU_Net','TransattU_Net','TransUNet'
LOSS_NAMES = losses.__all__  # 'BCEDiceLoss', 'LovaszHingeLoss'
LOSS_NAMES.append('BCEWithLogitsLoss')

"""

指定参数：
--dataset dsb2018_96 
--arch Dual_ANet

"""


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=4, type=int,
                        metavar='N', help='mini-batch size (default: 16)')

    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='DualA_Net',  # 修改模型的名字
                        choices=ARCH_NAMES,
                        help='model architecture: ' +
                             ' | '.join(ARCH_NAMES) +
                             ' (default: Trans4)')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)  # 是否使用深监督
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=96, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=96, type=int,
                        help='image height')

    # loss
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                             ' | '.join(LOSS_NAMES) +
                             ' (default: BCEDiceLoss)')

    # dataset
    parser.add_argument('--dataset', default='dsb2018_96',
                        help='dataset name')
    parser.add_argument('--img_ext', default='.png',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.png',
                        help='mask file extension')

    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                             ' | '.join(['Adam', 'SGD']) +
                             ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-3, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2 / 3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,   #=-1
                        metavar='N', help='early stopping (default: -1)')

    parser.add_argument('--num_workers', default=0, type=int)

    config = parser.parse_args()

    return config


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter(),
                  'jaccard': AverageMeter(),
                  # 'ACC': AverageMeter(),
                  'SE':AverageMeter(),
                  # 'SP':AverageMeter(),
                  'pre':AverageMeter(),
                  # 'F1':AverageMeter(),
                  }

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        # input = input.cuda()
        input = input.to(device)
        # target = target.cuda()
        target = target.to(device)

        # compute output
        if config['deep_supervision']:
            outputs = model(input)

            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            iou = iou_score(outputs[-1], target)
            dice = dice_coef(outputs[-1], target)
            jaccard = jac(outputs[-1], target)
            # ACC = get_acc(outputs[-1], target)
            SE = get_sensitivity(outputs[-1], target)
            # SP = get_specificity(outputs[-1], target)
            pre = get_precision(outputs[-1], target)
            # F1 = get_F1(outputs[-1], target)

        else:
            output = model(input)
            loss = criterion(output, target)
            iou = iou_score(output, target)
            dice = dice_coef(output, target)
            jaccard = jac(output, target)
            # ACC = get_acc(output, target)
            SE = get_sensitivity(output, target)
            # SP = get_specificity(output, target)
            pre = get_precision(output, target)
            # F1 = get_F1(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))  # input.size(0)指的是batch_size的值
        avg_meters['iou'].update(iou, input.size(0))
        avg_meters['dice'].update(dice, input.size(0))
        avg_meters['jaccard'].update(jaccard, input.size(0))
        # avg_meters['ACC'].update(ACC, input.size(0))
        avg_meters['SE'].update(SE, input.size(0))
        # avg_meters['SP'].update(SP, input.size(0))
        avg_meters['pre'].update(pre, input.size(0))
        # avg_meters['F1'].update(F1, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
            ('dice', avg_meters['dice'].avg),
            ('jaccard', avg_meters['jaccard'].avg),
            # ('ACC', avg_meters['ACC'].avg),
            ('SE', avg_meters['SE'].avg),
            # ('SP', avg_meters['SP'].avg),
            ('pre', avg_meters['pre'].avg),
            # ('F1', avg_meters['F1'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg),
                        ('jaccard', avg_meters['jaccard'].avg),
                        # ('ACC', avg_meters['ACC'].avg),
                        ('SE', avg_meters['SE'].avg),
                        # ('SP', avg_meters['SP'].avg),
                        ('pre', avg_meters['pre'].avg),
                        # ('F1', avg_meters['F1'].avg),
                        ])


def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter(),
                  'jaccard': AverageMeter(),
                  # 'ACC': AverageMeter(),
                  'SE': AverageMeter(),
                  # 'SP': AverageMeter(),
                  'pre': AverageMeter(),
                  # 'F1': AverageMeter(),
                  }

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            # input = input.cuda()
            input = input.to(device)
            # target = target.cuda()
            target = target.to(device)

            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou = iou_score(outputs[-1], target)
                dice = dice_coef(outputs[-1], target)
                jaccard = jac(outputs[-1], target)
                # ACC = get_acc(outputs[-1], target)
                SE = get_sensitivity(outputs[-1], target)
                # SP = get_specificity(outputs[-1], target)
                pre = get_precision(outputs[-1], target)
                # F1 = get_F1(outputs[-1], target)

            else:
                output = model(input)

                loss = criterion(output, target)
                iou = iou_score(output, target)
                dice = dice_coef(output, target)
                jaccard = jac(output, target)
                # ACC = get_acc(output, target)
                SE = get_sensitivity(output, target)
                # SP = get_specificity(output, target)
                pre = get_precision(output, target)
                # F1 = get_F1(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))
            avg_meters['jaccard'].update(jaccard, input.size(0))
            # avg_meters['ACC'].update(ACC, input.size(0))
            avg_meters['SE'].update(SE, input.size(0))
            # avg_meters['SP'].update(SP, input.size(0))
            avg_meters['pre'].update(pre, input.size(0))
            # avg_meters['F1'].update(F1, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg),
                ('jaccard', avg_meters['jaccard'].avg),
                # ('ACC', avg_meters['ACC'].avg),
                ('SE', avg_meters['SE'].avg),
                # ('SP', avg_meters['SP'].avg),
                ('pre', avg_meters['pre'].avg),
                # ('F1', avg_meters['F1'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg),
                        ('jaccard', avg_meters['jaccard'].avg),
                        # ('ACC', avg_meters['ACC'].avg),
                        ('SE', avg_meters['SE'].avg),
                        # ('SP', avg_meters['SP'].avg),
                        ('pre', avg_meters['pre'].avg),
                        # ('F1', avg_meters['F1'].avg),
                        ])


def main():
    config = vars(parse_args())

    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])
    os.makedirs('models/%s' % config['name'], exist_ok=True)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open('models/%s/config.yml' % config['name'], 'w') as f:  # 向yml写入config内容
        yaml.dump(config, f)

    # define loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':
        # criterion = nn.BCEWithLogitsLoss().cuda()
        criterion = nn.BCEWithLogitsLoss().to(device)  # WithLogits 就是先将输出结果经过sigmoid再交叉熵
    else:
        # criterion = losses.__dict__[config['loss']]().cuda()
        criterion = losses.__dict__[config['loss']]().to(device)
    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])

    # model = model.cuda()
    model = model.to(device)
    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')],
                                             gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    # Data loading code
    img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    # 随机划分测试集和训练集，test_size就是样本占比，random_state设置随机数种子
    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.3, random_state=41)
    # 数据增强：
    train_transform = Compose([
        # transforms.RandomRotate90(),
        albu.RandomRotate90(),  # 随机旋转90度
        albu.Flip(),  # 水平，垂直或水平和垂直翻转输入
        OneOf([
            transforms.HueSaturationValue(),  # 色调饱和度值
            transforms.RandomBrightness(),  # 随机亮度
            transforms.RandomContrast(),  # 随机对比度
        ], p=1),  # 按照归一化的概率选择执行哪一个
        albu.Resize(config['input_h'], config['input_w']),   #albumentations中的resize是重新调整图像大小
        transforms.Normalize(),
    ])

    val_transform = Compose([
        albu.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=train_transform)
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)  # 不能整除的batch是否就不要了
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('dice', []),
        ('jaccard', []),
        # ('ACC', []),
        ('SE',[]),
        # ('SP', []),
        ('pre', []),
        # ('F1', []),

        ('val_loss', []),
        ('val_iou', []),
        ('val_dice', []),
        ('val_jaccard', []),
        # ('val_ACC', []),
        ('val_SE', []),
        # ('val_SP', []),
        ('val_pre', []),
        # ('val_F1', []),

    ])

    best_iou = 0
    trigger = 0
    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print(
            'loss %.4f - iou %.4f - dice %.4f - jaccard %.4f - SE %.4f - pre %.4f - val_loss %.4f - val_iou %.4f - val_dice %.4f - val_jaccard %.4f - val_SE %.4f - val_pre %.4f'
            % (train_log['loss'], train_log['iou'], train_log['dice'], train_log['jaccard'],  train_log['SE'], train_log['pre'],
               val_log['loss'], val_log['iou'], val_log['dice'], val_log['jaccard'], val_log['SE'], val_log['pre']))

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['dice'].append(train_log['dice'])
        log['jaccard'].append(train_log['jaccard'])
        # log['ACC'].append(train_log['ACC'])
        log['SE'].append(train_log['SE'])
        # log['SP'].append(train_log['SP'])
        log['pre'].append(train_log['pre'])
        # log['F1'].append(train_log['F1'])

        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])
        log['val_jaccard'].append(val_log['jaccard'])
        # log['val_ACC'].append(val_log['ACC'])
        log['val_SE'].append(val_log['SE'])
        # log['SP'].append(val_log['SP'])
        log['val_pre'].append(val_log['pre'])
        # log['F1'].append(val_log['F1'])

        pd.DataFrame(log).to_csv('models/%s/log.csv' %
                                 config['name'], index=False)

        trigger += 1

        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), 'models/%s/model.pth' %
                       config['name'])
            best_iou = val_log['iou']
            print("=> saved best model")
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()  # 释放缓存


if __name__ == '__main__':
    main()
