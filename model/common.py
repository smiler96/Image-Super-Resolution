import torch.nn as nn
import math

def conv_(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

class BaseBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, bn=False, act=nn.ReLU(True)):
        super(BaseBlock, self).__init__()

        self.op = []
        self.op.append(conv_(in_c, out_c, kernel_size=kernel_size, stride=stride))
        if bn:
            self.op.append(nn.BatchNorm2d(out_c))
        if act is not  None:
            self.op.append(act)

        self.op = nn.Sequential(*self.op)

    def forward(self, x):
        x = self.op(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, n_feats=64, kernel_size=3, stride=1, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()

        assert act is not None

        self.res_scale = res_scale
        self.op = []
        for i in range(2):
            self.op.append(conv_(n_feats, n_feats, kernel_size=kernel_size, stride=stride, bias=bias))
            if bn:
                self.op.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                self.op.append(act)

        self.op = nn.Sequential(*self.op)

    def forward(self, x):
        res = self.op(x)
        res = res * self.res_scale

        x = res + x
        return x

class UpsampleBlock(nn.Module):
    def __init__(self, scale=4, n_feats=64, kernel_size=3, stride=1, bias=True, bn=False, act=nn.ReLU(True)):
        super(UpsampleBlock, self).__init__()

        self.op = []
        if (scale & (scale - 1)) == 0:
            for i in range(int(math.log2(scale))):
                self.op.append(conv_(n_feats, 4 * n_feats, kernel_size=kernel_size, stride=stride, bias=bias))
                self.op.append(nn.PixelShuffle(2))
                if bn:
                    self.op.append(nn.BatchNorm2d(n_feats))
                if act is not None:
                    self.op.append(act)
        elif(scale == 3):
            self.op.append(conv_(n_feats, 9 * n_feats, kernel_size=kernel_size, stride=stride, bias=bias))
            self.op.append(nn.PixelShuffle(3))
            if bn:
                self.op.append(nn.BatchNorm2d(n_feats))
            if act is not None:
                self.op.append(act)
        else:
            raise NotImplementedError

        self.op = nn.Sequential(*self.op)

    def forward(self, x):
        x = self.op(x)
        return x

class SELayer(nn.Module):
    def __init__(self, channels, reduction=16, act=nn.ReLU(True)):
        super(SELayer, self).__init__()

        self.op = nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            conv_(in_channels=channels, out_channels=channels // reduction, kernel_size=1, stride=1, padding=0,
                  bias=True),
            act,
            conv_(in_channels=channels // reduction, out_channels=channels, kernel_size=1, stride=1, padding=0,
                  bias=True),
            nn.Sigmoid(),
        ])

    def forward(self, x):
        s = self.op(x)
        x = x * s
        return x