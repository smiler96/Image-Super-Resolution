import torch
import torch.nn as nn
from model.common import conv_
from model.ddbpn import DPU

def wrapper(args):

    return DBPN_MR(in_c=args.n_colors, out_c=args.n_colors, scale=args.scale, n0=args.n_feats,
                 nr=args.nr, n_depths=args.n_depths)

class DBPN_MR(nn.Module):
    def __init__(self, in_c=3, out_c=3, scale=4, n0=128, nr=32, n_depths=6, n_iters=3, global_res=False):
        super(DBPN_MR, self).__init__()

        self.head = nn.Sequential(*[
            conv_(in_c, n0),
            nn.PReLU(n0),
            conv_(n0, nr, 1, 1, 0),
            nn.PReLU(nr),
        ])

        self.n_iters = n_iters
        self.global_res = global_res
        self.scale = scale

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

        self.tail = conv_(nr * n_iters, out_c)

    def forward(self, x):
        if self.global_res:
            x0 = nn.Upsample(scale_factor=self.scale, mode='bicubic')(x)
        x = self.head(x)

        # iterative recurrent
        res = []
        _d = x
        for itr in range(self.n_iters):
            ups_itr = []
            downs_itr = []

            # perform up-down projections
            for i in range(self.depths - 1):
                if i == 0:
                    _d = x
                else:
                    _d = torch.cat(downs_itr, dim=1)

                # up projection op
                _up = self.upmodules[i](_d)
                ups_itr.append(_up)

                # down projection op
                _d = self.downmodules[i](torch.cat(ups_itr, dim=1))
                downs_itr.append(_d)

            # back _d to the first up projection unit
            if itr != 0:
                x = _d

            # perform last up projection
            _up = self.upmodules[-1](torch.cat(downs_itr, dim=1))
            # ups_itr.append(_up)
            res.append(_up)

        x = self.tail(torch.cat(res, dim=1))
        if self.global_res:
            x = x + x0
        return x

if __name__ == "__main__":
    import numpy as np
    import torch
    import torchsummary

    model = DBPN_MR(in_c=3, out_c=3, scale=8, n0=128, nr=32, n_depths=6)
    print(torchsummary.summary(model, (3, 24, 24), device='cpu'))

    x = np.random.uniform(0, 1, [16, 3, 24, 24]).astype(np.float32)
    x = torch.tensor(x)

    # loss = nn.L1Loss()
    # Adam = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.99, 0.999))
    with torch.autograd.profiler.profile(use_cuda=False) as prof:
        y = model(x)
    print(prof)
    print(y.shape)
