import torch
from torch import nn
import numpy as np
import copy
import math
import paddle
import cv2
from einops import rearrange, repeat

__all__ = ['DualA_Net']


class ResNet_Block(nn.Module):
    def __init__(self, in_c, out_c, stride):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.BatchNorm2d(in_c),
            nn.ReLU(),
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(out_c),
        )

        self.attn = Squeeze_Excite_Block(out_c)

    def forward(self, inputs):
        x = self.c1(inputs)
        s = self.c2(inputs)
        y = self.attn(x + s)
        return y


class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)


class Squeeze_Excite_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Squeeze_Excite_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ASPP(nn.Module):
    def __init__(self, in_dims, out_dims, rate=[6, 12, 18]):  # 6 12 18 | 3 5 7 | 4 8 12
        super(ASPP, self).__init__()

        self.aspp_block1 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[0], dilation=rate[0]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block2 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[1], dilation=rate[1]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block3 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[2], dilation=rate[2]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )

        self.output = nn.Conv2d(len(rate) * out_dims, out_dims, 1)
        self._init_weights()

    def forward(self, x):
        x1 = self.aspp_block1(x)
        x2 = self.aspp_block2(x)
        x3 = self.aspp_block3(x)
        out = torch.cat([x1, x2, x3], dim=1)
        return self.output(out)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Upsample_(nn.Module):
    def __init__(self, scale=2):
        super(Upsample_, self).__init__()

        self.upsample = nn.Upsample(mode="bilinear", scale_factor=scale)

    def forward(self, x):
        return self.upsample(x)


class AttentionBlock(nn.Module):
    def __init__(self, input_decoder, input_encoder, output_dim):
        super(AttentionBlock, self).__init__()

        self.conv_encoder = nn.Sequential(
            nn.BatchNorm2d(input_encoder),
            nn.ReLU(),
            nn.Conv2d(input_encoder, output_dim, 3, padding=1),
            # nn.MaxPool2d(2, 2),
        )

        self.conv_decoder = nn.Sequential(
            nn.BatchNorm2d(input_decoder),
            nn.ReLU(),
            nn.Conv2d(input_decoder, output_dim, 3, padding=1),
        )

        # self.up1 = Upsample_()

        self.conv_attn = nn.Sequential(
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, 1, 1),
        )

    def forward(self, x1, x2):
        out = self.conv_encoder(x1) + self.conv_decoder(x2)
        out = self.conv_attn(out)
        return out * x2


# class SKConv(nn.Module):
#     def __init__(self, features, WH, M, G, r, stride=1, L=32):
#         """ Constructor
#         Args:
#             features: input channel dimensionality.
#             WH: input spatial dimensionality, used for GAP kernel size.
#             M: the number of branchs.
#             G: num of convolution groups.
#             r: the radio for compute d, the length of z.
#             stride: stride, default 1.
#             L: the minimum dim of the vector z in paper, default 32.
#         """
#         super(SKConv, self).__init__()
#         d = max(int(features / r), L)
#         self.M = M
#         self.features = features
#         self.convs = nn.ModuleList([])
#         for i in range(M):
#             self.convs.append(nn.Sequential(
#                 nn.Conv2d(features, features, kernel_size=3 + i * 2, stride=stride, padding=1 + i, groups=G),
#                 nn.BatchNorm2d(features),
#                 nn.ReLU(inplace=False)
#             ))
#         # self.gap = nn.AvgPool2d(int(WH/stride))
#         self.fc = nn.Linear(features, d)
#         self.fcs = nn.ModuleList([])
#         for i in range(M):
#             self.fcs.append(
#                 nn.Linear(d, features)
#             )
#         self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, x):
#         for i, conv in enumerate(self.convs):
#             fea = conv(x).unsqueeze_(dim=1)
#             if i == 0:
#                 feas = fea
#             else:
#                 feas = torch.cat([feas, fea], dim=1)
#         fea_U = torch.sum(feas, dim=1)
#         # fea_s = self.gap(fea_U).squeeze_()
#         fea_s = fea_U.mean(-1).mean(-1)
#         fea_z = self.fc(fea_s)
#         for i, fc in enumerate(self.fcs):
#             vector = fc(fea_z).unsqueeze_(dim=1)
#             if i == 0:
#                 attention_vectors = vector
#             else:
#                 attention_vectors = torch.cat([attention_vectors, vector], dim=1)
#         attention_vectors = self.softmax(attention_vectors)
#         attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
#         fea_v = (feas * attention_vectors).sum(dim=1)
#         return fea_v


class SKConv(nn.Module):
    def __init__(self, features, M=2, G=32, r=16, stride=1, L=32):
        super(SKConv, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3, stride=stride, padding=1 + i, dilation=1 + i, groups=G,
                          bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True)
            ))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Conv2d(features, d, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(d),
                                nn.ReLU(inplace=True))
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Conv2d(d, features, kernel_size=1, stride=1)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        batch_size = x.shape[0]  # 获取batch_size

        feats = [conv(x) for conv in self.convs]  # 让x分成3*3和5*5进行卷积
        feats = torch.cat(feats, dim=1)  # 合并卷积结果
        feats = feats.view(batch_size, self.M, self.features, feats.shape[2], feats.shape[3])
        # reshape一下大小
        # 接下来计算图中的U
        feats_U = torch.sum(feats, dim=1)  # 两个分支得到的卷积结果相加
        feats_S = self.gap(feats_U)  # 自适应池化，也就是对各个chanel求均值得到图中的S
        feats_Z = self.fc(feats_S)  # fc层压缩特征得到图中的Z

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        # 不同的分支各自恢复特征Z到channel的宽度
        attention_vectors = torch.cat(attention_vectors, dim=1)  # 连接起来方便后续操作
        attention_vectors = attention_vectors.view(batch_size, self.M, self.features, 1, 1)
        # reshape起来方便后续操作
        attention_vectors = self.softmax(attention_vectors)  # softmax得到图中的a和b

        feats_V = torch.sum(feats * attention_vectors, dim=1)
        # 把softmax后的各自自注意力跟卷积后的结果相乘，得到图中select的结果，然后相加得到最终输出

        return feats_V


class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)

        return x


# 软注意力机制 相当于是一个筛子 对encoder的当前层进行筛选
class AttentionGate(nn.Module):
    # 因为我们是将encoder的当前层 以及decoder的下一层上采样之后的结果送入Attention_Block的
    # 所以他们的尺寸以及通道数适相同的
    def __init__(self, in_channels, out_channels):
        super(AttentionGate, self).__init__()

        self.w_g = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels)
        )
        self.w_x = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.LeakyReLU()
        self.psi = nn.Sequential(
            nn.Conv2d(out_channels, 1, 1, 1, 0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi



class DualA_Netle):
    def __init__(self, in_c, out_c, n_classes=1, filters=[3, 16, 32, 64, 128]):
        super(DualA_Net, self).__init__()

        self.c1 = ResNet_Block(3, 16, stride=1)
        self.c2 = ResNet_Block(16, 32, stride=2)
        self.c3 = ResNet_Block(32, 64, stride=2)
        self.c4 = ResNet_Block(64, 128, stride=2)
        self.c5 = ResNet_Block(128, 256, stride=2)

        self.aspp = ASPP(16, 16)
        self.output = nn.Conv2d(16, 1, kernel_size=1, padding=0)
        # self.gap = nn.AdaptiveAvgPool3d(1, 96, 96)

        # self.a1 = AttentionBlock(128, 128, 128)
        # self.a2 = AttentionBlock(64, 64, 64)
        # self.a3 = AttentionBlock(32, 32, 32)
        # self.a4 = AttentionBlock(16, 16, 16)

        self.attention_gate1 = AttentionGate(128, 32)
        self.attention_gate2 = AttentionGate(64, 32)
        self.attention_gate3 = AttentionGate(32, 32)
        self.attention_gate4 = AttentionGate(16, 32)


        self.down = nn.MaxPool2d(2, 2, ceil_mode=True)

        ## -------------Decoder--------------
        # self.CatChannels = filters[0]
        # self.CatBlocks = 5
        # self.UpChannels = self.CatChannels * self.CatBlocks

        '''stage 4d'''

        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = nn.Conv2d(64, 128, 3, padding=1)
        self.h3_PT_hd4_bn = nn.BatchNorm2d(128)
        self.h3_PT_hd4_relu = nn.ReLU(inplace=True)

        # h4->40*40, hd4->40*40, Concatenation
        self.h4_Cat_hd4_conv = nn.Conv2d(128, 128, 3, padding=1)
        self.h4_Cat_hd4_bn = nn.BatchNorm2d(128)
        self.h4_Cat_hd4_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->40*40, Upsample 2 times
        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd5_UT_hd4_conv = nn.Conv2d(256, 128, 3, padding=1)
        self.hd5_UT_hd4_bn = nn.BatchNorm2d(128)
        self.hd5_UT_hd4_relu = nn.ReLU(inplace=True)

        # h1->80*80, hd4->40*40, Pooling 2 times
        self.h1_PT_hd4 = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv = nn.Conv2d(16, 128, 3, padding=1)
        self.h1_PT_hd4_bn = nn.BatchNorm2d(128)
        self.h1_PT_hd4_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        # #特征聚合机制
        self.conv4d_1 = nn.Conv2d(384, 128, 3, padding=1)  # 16
        self.bn4d_1 = nn.BatchNorm2d(128)
        self.relu4d_1 = nn.ReLU(inplace=True)

        '''stage 3d'''

        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = nn.Conv2d(32, 64, 3, padding=1)
        self.h2_PT_hd3_bn = nn.BatchNorm2d(64)
        self.h2_PT_hd3_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_conv = nn.Conv2d(64, 64, 3, padding=1)
        self.h3_Cat_hd3_bn = nn.BatchNorm2d(64)
        self.h3_Cat_hd3_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd4->80*80, Upsample 2 times
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd4_UT_hd3_conv = nn.Conv2d(128, 64, 3, padding=1)
        self.hd4_UT_hd3_bn = nn.BatchNorm2d(64)
        self.hd4_UT_hd3_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = nn.Conv2d(192, 64, 3, padding=1)  # 16
        self.bn3d_1 = nn.BatchNorm2d(64)
        self.relu3d_1 = nn.ReLU(inplace=True)

        '''stage 2d '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = nn.Conv2d(16, 32, 3, padding=1)
        self.h1_PT_hd2_bn = nn.BatchNorm2d(32)
        self.h1_PT_hd2_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_conv = nn.Conv2d(32, 32, 3, padding=1)
        self.h2_Cat_hd2_bn = nn.BatchNorm2d(32)
        self.h2_Cat_hd2_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd3_UT_hd2_conv = nn.Conv2d(64, 32, 3, padding=1)
        self.hd3_UT_hd2_bn = nn.BatchNorm2d(32)
        self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1 = nn.Conv2d(96, 32, 3, padding=1)  # 16
        self.bn2d_1 = nn.BatchNorm2d(32)
        self.relu2d_1 = nn.ReLU(inplace=True)

        '''stage 1d'''
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv = nn.Conv2d(16, 16, 3, padding=1)
        self.h1_Cat_hd1_bn = nn.BatchNorm2d(16)
        self.h1_Cat_hd1_relu = nn.ReLU(inplace=True)

        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd2_UT_hd1_conv = nn.Conv2d(32, 16, 3, padding=1)
        self.hd2_UT_hd1_bn = nn.BatchNorm2d(16)
        self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = nn.Conv2d(32, 16, 3, padding=1)  # 16
        self.bn1d_1 = nn.BatchNorm2d(16)
        self.relu1d_1 = nn.ReLU(inplace=True)


        # -------------Bilinear Upsampling--------------
        self.upscore6 = nn.Upsample(scale_factor=32, mode='bilinear')  ###
        self.upscore5 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upscore3 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.sk1 = SKConv(128)
        self.sk2 = SKConv(64)
        self.sk3 = SKConv(32)


    def forward(self, inputs):
        c1 = self.c1(inputs)
        c2 = self.c2(c1)
        c3 = self.c3(c2)
        c4 = self.c4(c3)
        c5 = self.c5(c4)

        # b1 = self.b1(c5)

        ## -------------Decoder------------
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(c3))))
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(c4)))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(c5))))
        hd1_ADD_hd4 = self.sk1(self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(c1)))))
        hd4 = self.sk1(self.attention_gate1(hd1_ADD_hd4, self.relu4d_1(self.bn4d_1(self.conv4d_1(
            torch.cat((h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1))))))  # hd4->40*40*UpChannels

        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(c2))))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(c3)))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))))
        hd3 = self.sk2(self.relu3d_1(self.bn3d_1(self.conv3d_1(
            torch.cat((h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3), 1)))))  # hd3->80*80*UpChannels

        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(c1))))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(c2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))))
        hd2 = self.sk3(self.relu2d_1(self.bn2d_1(self.conv2d_1(
            torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2), 1)))))  # hd2->160*160*UpChannels

        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(c1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))))
        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(torch.cat((hd2_UT_hd1, h1_Cat_hd1), 1))))  # hd1->320*320*UpChannels
        # print(hd1.shape)

        output = self.aspp(hd1)
        output = self.output(output)
        # output = self.gap(output)

        return output
