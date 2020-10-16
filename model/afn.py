import torch
import torch.nn as nn
from model.common import UpsampleBlock, conv_, SELayer

def wrapper(args):
    act = None
    if args.act == 'relu':
        act = nn.ReLU(True)
    elif args.act == 'leak_relu':
        act = nn.LeakyReLU(0.2, True)
    elif args.act is None:
        act = None
    else:
        raise NotImplementedError

    return AFN(in_c=args.n_colors, out_c=args.n_colors, scale=args.scale, n_feats=args.n_feats, act=act)

class AFB_0(nn.Module):
    def __init__(self, channels, n_blocks=2, act=nn.ReLU(True)):
        super(AFB_0, self).__init__()
        self.op = []
        for _ in range(n_blocks):
            self.op.append(conv_(channels, channels))
            self.op.append(act)

        self.op = nn.Sequential(*self.op)

    def forward(self, x):
        x = x + self.op(x)
        return x


class AFB_L1(nn.Module):
    def __init__(self, channels, n_l0=3, act=nn.ReLU(True)):
        super(AFB_L1, self).__init__()

        self.n = n_l0
        self.convs_ = nn.ModuleList()
        for _ in range(n_l0):
            self.convs_.append(
                AFB_0(channels, 2, act)
            )

        self.LFF = nn.Sequential(
            SELayer(channels * n_l0, 16),
            nn.Conv2d(channels * n_l0, channels, 1, padding=0, stride=1),
        )

    def forward(self, x):
        res = []
        ox = x

        for i in range(self.n):
            x = self.convs_[i](x)
            res.append(x)
        res = self.LFF(torch.cat(res, 1))
        x = res + ox
        return x


class AFB_L2(nn.Module):
    def __init__(self, channels, n_l1=4, act=nn.ReLU(True)):
        super(AFB_L2, self).__init__()

        self.n = n_l1
        self.convs_ = nn.ModuleList()
        for _ in range(n_l1):
            self.convs_.append(
                AFB_L1(channels, 3, act)
            )

        self.LFF = nn.Sequential(
            SELayer(channels * n_l1, 16),
            nn.Conv2d(channels * n_l1, channels, 1, padding=0, stride=1),
        )

    def forward(self, x):
        res = []
        ox = x

        for i in range(self.n):
            x = self.convs_[i](x)
            res.append(x)
        res = self.LFF(torch.cat(res, 1))
        x = res + ox
        return x


class AFB_L3(nn.Module):
    def __init__(self, channels, n_l2=4, act=nn.ReLU(True)):
        super(AFB_L3, self).__init__()

        self.n = n_l2
        self.convs_ = nn.ModuleList()
        for _ in range(n_l2):
            self.convs_.append(
                AFB_L2(channels, 4, act)
            )

        self.LFF = nn.Sequential(
            SELayer(channels * n_l2, 16),
            nn.Conv2d(channels * n_l2, channels, 1, padding=0, stride=1),
        )

    def forward(self, x):
        res = []
        ox = x

        for i in range(self.n):
            x = self.convs_[i](x)
            res.append(x)
        res = self.LFF(torch.cat(res, 1))
        x = res + ox
        return x


class AFN(nn.Module):
    def __init__(self, in_c=3, out_c=3, scale=4, n_feats=128, n_l3=3, act=nn.LeakyReLU(0.2, True)):
        super(AFN, self).__init__()

        self.head = conv_(in_c, n_feats)

        self.n = n_l3
        self.AFBs = nn.ModuleList()
        for i in range(n_l3):
            self.AFBs.append(
                AFB_L3(channels=n_feats, n_l2=4, act=act)
            )

        self.GFF = nn.Sequential(*[
            SELayer(n_feats * n_l3),
            conv_(n_feats * n_l3, n_feats, 1, padding=0, stride=1),
        ])

        self.tail = nn.Sequential(*[
            UpsampleBlock(scale, n_feats, kernel_size=3, stride=1, bias=True, act=act),
            conv_(n_feats, out_c)
        ])

    def forward(self, x):
        res = []
        x = self.head(x)

        for i in range(self.n):
            x = self.AFBs[i](x)
            res.append(x)

        res = self.GFF(torch.cat(res, 1))
        x = res + x

        x = self.tail(x)
        return x

if __name__ == "__main__":
    import numpy as np
    import torch
    import torchsummary

    model = AFN(in_c=3, out_c=3, scale=8, n_feats=128, n_l3=3, act=nn.LeakyReLU(0.2, True))
    print(torchsummary.summary(model, (3, 24, 24), device='cpu'))

    x = np.random.uniform(0, 1, [2, 3, 24, 24]).astype(np.float32)
    x = torch.tensor(x)

    # loss = nn.L1Loss()
    # Adam = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.99, 0.999))
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        y = model(x)
    print(prof)
    print(y.shape)
