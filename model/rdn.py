import torch
import torch.nn as nn
from model.common import UpsampleBlock, conv_

def wrapper(args):

    return RDN(in_c=args.n_colors, out_c=args.n_colors, scale=args.scale, n_feats=args.n_feats,
               D=args.D, G=args.G, C=args.C)

class ConvLayer(nn.Module):
    def __init__(self, in_c, out_c):
        super(ConvLayer, self).__init__()
        self.conv_ = nn.Sequential(*[
            conv_(in_c, out_c, kernel_size=3, stride=1, bias=True),
            nn.ReLU(True),
        ])

    def forward(self, x):
        x = self.conv_(x)
        return x


class ResDenseBlock(nn.Module):
    def __init__(self, G0=64, G=32, C=6):
        super(ResDenseBlock, self).__init__()

        self.C = C
        self.convs_ = nn.ModuleList()
        for i in range(C):
            self.convs_.append(
                ConvLayer(G0+i*G, G)
            )

        self.fusion = conv_(in_channels=G0+C*G, out_channels=G0, kernel_size=1, stride=1,
                            padding=0, bias=True)

    def forward(self, x):
        res = []
        res.append(x)
        for i in range(self.C):
            llf = torch.cat(res, dim=1)
            llf = self.convs_[i](llf)
            res.append(llf)

        llf = torch.cat(res, dim=1)
        res = self.fusion(llf)

        x = res + x
        return x


class RDN(nn.Module):
    def __init__(self, in_c=3, out_c=3, scale=4, n_feats=64, D=20, G=32, C=6):
        super(RDN, self).__init__()

        self.SFE1 = conv_(in_c, n_feats, 3, 1, bias=True)
        self.SFE2 = conv_(n_feats, n_feats, 3, 1, bias=True)

        self.D = D
        self.RDNs_ = nn.ModuleList()
        for i in range(D):
            self.RDNs_.append(
                ResDenseBlock(G0=n_feats, G=G, C=C)
            )

        self.GFF = nn.Sequential(*[
            conv_(in_channels=n_feats * D, out_channels=n_feats, kernel_size=1, stride=1,
                  padding=0, bias=True),
            conv_(n_feats, n_feats)
        ])

        self.tail = nn.Sequential(*[
            UpsampleBlock(scale, n_feats, 3, 1, True, False),
            conv_(n_feats, out_c, kernel_size=3, stride=1, bias=True),
        ])

    def forward(self, x):
        F_1 = self.SFE1(x)
        x = self.SFE2(F_1)

        dff = []
        for i in range(self.D):
            if i==0:
                rdb_i = self.RDNs_[i](x)
            else:
                rdb_i = self.RDNs_[i](dff[i-1])
            dff.append(rdb_i)

        dff = torch.cat(dff, dim=1)
        dff = self.GFF(dff)

        x = F_1 + dff
        x = self.tail(x)
        return x

if __name__ == "__main__":
    import numpy as np
    import torch
    import torchsummary

    model = RDN(in_c=3, out_c=3, scale=8, n_feats=64, D=20, G=32, C=6)
    print(torchsummary.summary(model, (3, 24, 24), device='cpu'))

    x = np.random.uniform(0, 1, [16, 3, 24, 24]).astype(np.float32)
    x = torch.tensor(x)

    # loss = nn.L1Loss()
    # Adam = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.99, 0.999))
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        y = model(x)
    print(prof)
    print(y.shape)
