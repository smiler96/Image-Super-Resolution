import torch.nn as nn

def wrapper(args):

    return DDBPN(in_c=args.n_colors, out_c=args.n_colors, scale=args.scale, n_feats=args.n_feats,
                 n_dense=args.n_dense)

class ConvLayer(nn.Module):
    def __init__(self, in_c, out_c, scale=8, up=True):
        super(ConvLayer, self).__init__()
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

        self.op1 = ConvLayer(n_feats, nr, scale, up)
        self.op2 = ConvLayer(nr, n_feats, scale, not up)
        self.op3 = ConvLayer(n_feats, nr, scale, up)

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
    def __init__(self, in_c=3, out_c=3, scale=4, n_feats=64, n_dense=6):
        super(DDBPN, self).__init__()

    def forward(self, x):
        return x

if __name__ == "__main__":
    import numpy as np
    import torch
    import torchsummary

    model = DDBPN(in_c=3, out_c=3, scale=8, n_feats=128, n_dense=6)
    print(torchsummary.summary(model, (3, 24, 24), device='cpu'))

    x = np.random.uniform(0, 1, [16, 3, 24, 24]).astype(np.float32)
    x = torch.tensor(x)

    # loss = nn.L1Loss()
    # Adam = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.99, 0.999))
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        y = model(x)
    print(prof)
    print(y.shape)
