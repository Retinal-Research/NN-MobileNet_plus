"""
ReXNet
Copyright (c) 2020-present NAVER Corp.
MIT license
"""

import torch
import torch.nn as nn
from math import ceil
from model.S3_DSConv_pro import DSConv_pro
from model.vit_block import MobileViTBlock, conv_1x1_bn, MV2Block
# from vit_block import MobileViTBlockv2

USE_MEMORY_EFFICIENT_SiLU = True

if USE_MEMORY_EFFICIENT_SiLU:
    @torch.jit.script
    def silu_fwd(x):
        return x.mul(torch.sigmoid(x))


    @torch.jit.script
    def silu_bwd(x, grad_output):
        x_sigmoid = torch.sigmoid(x)
        return grad_output * (x_sigmoid * (1. + x * (1. - x_sigmoid)))


    class SiLUJitImplementation(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return silu_fwd(x)

        @staticmethod
        def backward(ctx, grad_output):
            x = ctx.saved_tensors[0]
            return silu_bwd(x, grad_output)


    def silu(x, inplace=False):
        return SiLUJitImplementation.apply(x)

else:
    def silu(x, inplace=False):
        return x.mul_(x.sigmoid()) if inplace else x.mul(x.sigmoid())


class SiLU(nn.Module):
    def __init__(self, inplace=True):
        super(SiLU, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return silu(x, self.inplace)


def ConvBNAct(out, in_channels, channels, kernel=1, stride=1, pad=0,
              num_group=1, active=True, relu6=False):
    out.append(nn.Conv2d(in_channels, channels, kernel,
                         stride, pad, groups=num_group, bias=False))
    out.append(nn.BatchNorm2d(channels))
    if active:
        out.append(nn.ReLU6(inplace=True) if relu6 else nn.ReLU(inplace=True))

def DSConvBN(out,
             in_channels,
             out_channels,
             kernel_size=9,
             morph=0,               
             extend_scope=1.0,
             if_offset=True,
             device='cuda'):
    out.append(DSConv_pro(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        extend_scope=extend_scope,
        morph=morph,
        if_offset=if_offset,
        device=device
    ))
    out.append(nn.BatchNorm2d(out_channels))

def ConvBNSiLU(out, in_channels, channels, kernel=1, stride=1, pad=0, num_group=1):
    out.append(nn.Conv2d(in_channels, channels, kernel,
                         stride, pad, groups=num_group, bias=False))
    out.append(nn.BatchNorm2d(channels))
    out.append(SiLU(inplace=True))

class Adapter(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.proj = nn.Conv2d(in_c, out_c, kernel_size=1, bias=False)
        self.bn   = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn(self.proj(x))

class LinearBottleneck(nn.Module):
    def __init__(self, in_channels, channels, t, stride, dim, use_vit=False, use_dsc = False, se_ratio=12, dropout = 0.0, L=2,
                 **kwargs):
        super(LinearBottleneck, self).__init__(**kwargs)
        self.use_shortcut = stride == 1 and in_channels <= channels
        self.in_channels = in_channels
        self.out_channels = channels
        self.drop = nn.Dropout2d(dropout)
        out = []
        if use_vit and L!=2:
            out.append(MV2Block(in_channels, channels, stride=stride, expansion=4))
        else:
            if t != 1:
                dw_channels = in_channels * t

                ConvBNSiLU(out, in_channels=in_channels, channels=dw_channels)
            else:
                dw_channels = in_channels

            if stride == 2 and use_dsc:
                DSConvBN(out, in_channels=dw_channels, out_channels=dw_channels, kernel_size=3, morph=0)
            else :
                ConvBNAct(out, in_channels=dw_channels, channels=dw_channels, kernel=3, stride=stride, pad=1,
                        num_group=dw_channels, active=False)

            out.append(nn.ReLU6())
            ConvBNAct(out, in_channels=dw_channels, channels=channels, active=False, relu6=True)
        
        if use_vit:
            if L == 2:
                out.append(Adapter(61, 64))                
                out.append(MobileViTBlock(dim, L, 64, 3, (2,2), int(dim * 2)))
            else:
                out.append(MobileViTBlock(dim, L, channels, 3, (2,2), int(dim * 2)))

        self.out = nn.Sequential(*out)

    def forward(self, x):
        out = self.out(x)
        if self.use_shortcut:
            out[:, 0:self.in_channels] += x
        out = self.drop(out)
        return out


class ReXNetV1(nn.Module):
    def __init__(self, input_ch=16, final_ch=180, width_mult=1.0, depth_mult=1.0, classes=1000,
                 use_se=True,
                 se_ratio=12,
                 dropout_ratio=0.2,
                 dropout_path = 0.25,
                 bn_momentum=0.9):
        super(ReXNetV1, self).__init__()

        # layers = [1, 2, 2, 3, 3, 5]
        # strides = [1, 2, 2, 2, 1, 2]
        layers = [1, 2, 2, 1, 1]
        strides = [1, 2, 2, 2, 2]

        dim = [
            0,        # stage 0: 1 block
            0, 0,     # stage 1: 2 blocks
            0, 96,   # stage 2: 2 blocks
            120,         # stage 3: 1 block
            144          # stage 4: 1 block
        ]

        L  = [
            0,        # stage 0: 1 block
            0, 0, # stage 1: 2 blocks
            0, 2,   # stage 2: 2 blocks
            4,         # stage 3: 1 block
            3          # stage 4: 1 block
        ]

        use_vit_blocks = [
            False,        # stage 0: 1 block
            False, False, # stage 1: 2 blocks
            False, True,   # stage 2: 2 blocks
            True,         # stage 3: 1 block
            True          # stage 4: 1 block
        ]
        layers = [ceil(element * depth_mult) for element in layers]
        strides = sum([[element] + [1] * (layers[idx] - 1)
                       for idx, element in enumerate(strides)], [])

            
        ts = [1] * layers[0] + [6] * sum(layers[1:])
        print(ts)
        self.depth = sum(layers[:])
        stem_channel = 32 / width_mult if width_mult < 1.0 else 32
        inplanes = input_ch / width_mult if width_mult < 1.0 else input_ch

        features = []
        in_channels_group = [32, 16, 27, 38, 50, 64, 80]
        channels_group = [16, 27, 38, 50, 61, 80, 96]
        # The following channel configuration is a simple instance to make each layer become an expand layer.
        # for i in range(self.depth):
        #     if i == 0:
        #         in_channels_group.append(int(round(stem_channel * width_mult)))
        #         channels_group.append(int(round(inplanes * width_mult)))
        #     else:
        #         in_channels_group.append(int(round(inplanes * width_mult)))
        #         inplanes += final_ch / (self.depth * 1.0)
        #         channels_group.append(int(round(inplanes * width_mult)))

        ConvBNSiLU(features, 3, int(round(stem_channel * width_mult)), kernel=3, stride=2, pad=1)
        print(in_channels_group)
        print(channels_group)
        for block_idx, (in_c, c, t, s, vit, d, l) in enumerate(zip(in_channels_group, channels_group, ts, strides, use_vit_blocks, dim, L)):

            features.append(LinearBottleneck(in_channels=in_c,
                                             channels=c,
                                             t=t,
                                             stride=s,
                                             dim= d,
                                             L = l,
                                             se_ratio=se_ratio, dropout = dropout_path, use_vit=vit, use_dsc=False if block_idx == 3 else False))

        # pen_channels = int(1280 * width_mult)
        # ConvBNSiLU(features, c, pen_channels)
        features.append(conv_1x1_bn(96, 384))

        features.append(nn.AdaptiveAvgPool2d(1))

        self.features = nn.Sequential(*features)
        # self.output = nn.Sequential(
        #     nn.Dropout(dropout_ratio),
        #     nn.Conv2d(pen_channels, classes, 1, bias=True))
        self.out = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Conv2d(384, classes, 1, bias=True))
    def extract_features(self, x):
        return self.features[:-1](x)
    
    def forward(self, x):
        x = self.features(x)
        x = self.out(x).flatten(1)
        return x

import torch

def inspect_checkpoint(ckpt_path, topk=30):
    """
    打印 checkpoint 里的 key 和 shape
    
    参数:
        ckpt_path (str): 权重文件路径
        topk (int): 打印前多少个参数，默认 30
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # 如果有 "model" key，就取出来
    if isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]

    print(f"✅ 总共有 {len(ckpt)} 个参数\n")
    for i, (k, v) in enumerate(ckpt.items()):
        print(f"{i:4d} | {k:50s} | {tuple(v.shape)}")
        if i + 1 >= topk:
            break