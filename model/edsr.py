import torch.nn as nn
from model.common import ResBlock, UpsampleBlock, conv_

def wrapper(args):
    act = None
    if args.act == 'relu':
        act = nn.ReLU(True)
    elif args.act is None:
        act = None
    else:
        raise NotImplementedError

    last_act = None
    if args.last_act is not None:
        if args.last_act == 'sigmoid':
            last_act = nn.Sigmoid()
        elif args.last_act == 'tanh':
            last_act = nn.Tanh()
        else:
            raise NotImplementedError

    return EDSR(in_c=args.n_colors, out_c=args.n_colors, scale=args.scale, n_feats=args.n_feats,
                n_resblocks=args.n_resblocks, kernel_size=args.kernel_size, stride=args.stride,
                bias=True, bn=args.bn, act=act, res_scale=args.res_scale, last_act=last_act)

class EDSR(nn.Module):
    def __init__(self, in_c=3, out_c=3, scale=4, n_feats=64, n_resblocks=16,
                 kernel_size=3, stride=1, bias=True, bn=False, act=nn.ReLU(True),
                 res_scale=1, last_act=None):
        super(EDSR, self).__init__()

        self.head = conv_(in_c, n_feats, kernel_size=kernel_size, stride=stride)

        self.body = [ResBlock(n_feats=n_feats, kernel_size=kernel_size, stride=stride, bias=bias,
                              bn=bn, act=act, res_scale=res_scale) for _ in range(n_resblocks)]
        self.body = nn.Sequential(*self.body)

        self.tail = [UpsampleBlock(scale=scale, n_feats=n_feats, kernel_size=kernel_size,
                                   stride=stride, bias=bias, bn=bn, act=act),
                     conv_(n_feats, out_c, kernel_size=kernel_size, stride=stride, bias=bias)]
        if last_act is not None:
            self.tail.append(nn.Sigmoid())
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

    model = EDSR(in_c=3, out_c=3, scale=4, n_feats=64, n_resblocks=16, bn=True)
    print(torchsummary.summary(model, (3, 48, 48), device='cpu'))

    x = np.random.uniform(0, 1, [2, 3, 48, 48]).astype(np.float32)
    x = torch.tensor(x)

    # loss = nn.L1Loss()
    # Adam = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.99, 0.999))
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        y = model(x)
    print(prof)
    print(y.shape)
