import torch
import torch.nn as nn
from model.common import conv_

def wrapper(args):

    return DDBPN(in_c=args.n_colors, out_c=args.n_colors, scale=args.scale, n0=args.n_feats,
                 nr=args.nr, n_depths=args.n_depths)

class ProjectConvLayer(nn.Module):
    def __init__(self, in_c, out_c, scale=8, up=True):
        super(ProjectConvLayer, self).__init__()
        k, s, p = {
            2: [6, 2, 2],
            4: [8, 4, 2],
            8: [12, 8, 2]
        }[scale]
        if up:
            self.conv_ = nn.Sequential(*[
                nn.ConvTranspose2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=True),
                nn.PReLU(num_parameters=out_c),
            ])
        else:
            self.conv_ = nn.Sequential(*[
                nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=True),
                nn.PReLU(num_parameters=out_c),
            ])

    def forward(self, x):
        x = self.conv_(x)
        return x


# dense projection unit
class DPU(nn.Module):
    def __init__(self, in_c=128, nr=32, scale=8, up=True, bottleneck=True):
        super(DPU, self).__init__()

        if bottleneck:
            self.bottleblock = nn.Conv2d(in_c, nr, kernel_size=1, stride=1, padding=0, bias=True)
            n_feats = nr
        else:
            self.bottleblock = None
            n_feats = in_c

        self.op1 = ProjectConvLayer(n_feats, nr, scale, up)
        self.op2 = ProjectConvLayer(nr, n_feats, scale, not up)
        self.op3 = ProjectConvLayer(n_feats, nr, scale, up)

    def forward(self, x):
        if self.bottleblock is not None:
            x = self.bottleblock(x)

        h_0 = self.op1(x)
        l_0 = self.op2(h_0)
        e = l_0 - x
        h_1 = self.op3(e)
        x = h_1 + h_0

        return x


class DDBPN(nn.Module):
    def __init__(self, in_c=3, out_c=3, scale=4, n0=128, nr=32, n_depths=6):
        super(DDBPN, self).__init__()

        self.head = nn.Sequential(*[
            conv_(in_c, n0),
            nn.PReLU(n0),
            conv_(n0, nr, 1, 1, 0),
            nn.PReLU(nr),
        ])

        self.depths = n_depths
        self.upmodules = nn.ModuleList()
        self.downmodules = nn.ModuleList()
        # up sample projection units
        chs = nr
        for i in range(n_depths):
            self.upmodules.append(
                DPU(chs, nr, scale, True, i != 0)
            )
            if i != 0:
                chs += nr
        # down sample projection units
        chs = nr
        for i in range(n_depths-1):
            self.downmodules.append(
                DPU(chs, nr, scale, False, i != 0)
            )
            chs += nr

        self.tail = conv_(nr * n_depths, out_c)

    def forward(self, x):
        x = self.head(x)

        ups = []
        downs = []
        for i in range(self.depths - 1):
            if i==0:
                _d = x
            else:
                _d = torch.cat(downs, dim=1)

            _up = self.upmodules[i](_d)
            ups.append(_up)
            _down = self.downmodules[i](torch.cat(ups, dim=1))
            downs.append(_down)

        _up = self.upmodules[-1](torch.cat(downs, dim=1))
        ups.append(_up)

        x = self.tail(torch.cat(ups, dim=1))
        return x

if __name__ == "__main__":
    import numpy as np
    import torch
    import torchsummary

    model = DDBPN(in_c=3, out_c=3, scale=8, n0=128, nr=32, n_depths=6)
    print(torchsummary.summary(model, (3, 24, 24), device='cpu'))

    x = np.random.uniform(0, 1, [16, 3, 24, 24]).astype(np.float32)
    x = torch.tensor(x)

    # loss = nn.L1Loss()
    # Adam = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.99, 0.999))
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        y = model(x)
    print(prof)
    print(y.shape)
