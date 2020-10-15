import torch.nn as nn
from model.common import UpsampleBlock, conv_

def wrapper(args):
    act = None
    if args.act == 'relu':
        act = nn.ReLU(True)
    elif args.act is None:
        act = None
    else:
        raise NotImplementedError

    return RCAN(in_c=args.n_colors, out_c=args.n_colors, scale=args.scale, n_feats=args.n_feats,
                n_rg=args.n_rg, n_rcab=args.n_rcab, kernel_size=args.kernel_size, stride=args.stride,
                bias=True, bn=args.bn, act=act)

class ChannelAttentation(nn.Module):
    def __init__(self, channels=64, reduction=16):
        super(ChannelAttentation, self).__init__()

        self.op = nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            conv_(in_channels=channels, out_channels=channels // reduction, kernel_size=1, stride=1, padding=0,
                  bias=True),
            nn.ReLU(inplace=True),
            conv_(in_channels=channels // reduction, out_channels=channels, kernel_size=1, stride=1, padding=0,
                  bias=True),
            nn.Sigmoid(),
        ])

    def forward(self, x):
        s = self.op(x)
        x = x * s
        return x

class ResChannelAttBlock(nn.Module):
    def __init__(self, n_feats=64, reduction=16, kernel_size=3, stride=1, bias=True, bn=False, instance_norm=False,
                 act=nn.ReLU(True)):
        super(ResChannelAttBlock, self).__init__()
        assert act is not None
        self.op = []
        for i in range(2):
            self.op.append(conv_(n_feats, n_feats, kernel_size=kernel_size, stride=stride, bias=bias))
            if bn:
                self.op.append(nn.BatchNorm2d(n_feats))
            if instance_norm:
                self.op.append(nn.InstanceNorm2d(n_feats))
            if i == 0:
                self.op.append(act)
        self.op.append(ChannelAttentation(channels=n_feats, reduction=reduction))
        self.op = nn.Sequential(*self.op)

    def forward(self, x):
        res = self.op(x)
        x = res + x
        return x

class ResGroup(nn.Module):
    def __init__(self, n_rcab=20, n_feats=64, reduction=16, kernel_size=3, stride=1, bias=True, bn=False,
                 instance_norm=False, act=nn.ReLU(True)):
        super(ResGroup, self).__init__()
        assert act is not None
        self.op = []

        for _ in range(n_rcab):
            self.op.append(ResChannelAttBlock(n_feats=n_feats, reduction=reduction, kernel_size=kernel_size,
                                              stride=stride, bias=bias, bn=bn, instance_norm=instance_norm, act=act))

        self.op.append(conv_(in_channels=n_feats, out_channels=n_feats, kernel_size=kernel_size, stride=stride, bias=bias))
        self.op = nn.Sequential(*self.op)

    def forward(self, x):
        res = self.op(x)
        x = res + x
        return x

class RCAN(nn.Module):
    def __init__(self, in_c=3, out_c=3, scale=4, n_feats=64, n_rg=10, n_rcab=20,
                 kernel_size=3, stride=1, bias=True, bn=False, instance_norm=False, act=nn.ReLU(True)):
        super(RCAN, self).__init__()

        self.head = conv_(in_c, n_feats, kernel_size=kernel_size, stride=stride)

        self.body = [ResGroup(n_rcab=n_rcab, n_feats=n_feats, kernel_size=kernel_size, stride=stride, bias=bias,
                              bn=bn, instance_norm=instance_norm, act=act) for _ in range(n_rg)]
        self.body = nn.Sequential(*self.body)

        self.tail = [UpsampleBlock(scale=scale, n_feats=n_feats, kernel_size=kernel_size,
                                   stride=stride, bias=bias, bn=bn, act=act)]
        if instance_norm:
            self.tail.append(nn.InstanceNorm2d(n_feats))
        self.tail.append(conv_(n_feats, out_c, kernel_size=kernel_size, stride=stride, bias=bias))
        self.tail = nn.Sequential(*self.tail)

    def forward(self, x):
        x = self.head(x)

        res = self.body(x)
        x = res + x

        x = self.tail(x)

        return x

if __name__ == "__main__":
    import numpy as np
    import torch
    import torchsummary

    model = RCAN(in_c=3, out_c=3, scale=8, n_feats=64, n_rg=10, n_rcab=20, bn=True)
    print(torchsummary.summary(model, (3, 16, 16), device='cpu'))

    x = np.random.uniform(0, 1, [2, 3, 16, 16]).astype(np.float32)
    x = torch.tensor(x)

    # loss = nn.L1Loss()
    # Adam = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.99, 0.999))
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        y = model(x)
    print(prof)
    print(y.shape)
