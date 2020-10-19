import torch
import torch.nn as nn
from model.common import UpsampleBlock, conv_
from model.rcan import ResGroup

def wrapper(args):

    return HAN(in_c=args.n_colors, out_c=args.n_colors, scale=args.scale, n_feats=args.n_feats,
                n_rg=args.n_rg, n_rcab=args.n_rcab,)

class LAM(nn.Module):
    def __init__(self):
        super(LAM, self).__init__()

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        bs, N, C, H, W = x.size()
        query = x.view(bs, N, -1)
        key = x.view(bs, N, -1).permute(0, 2, 1)
        energy = torch.bmm(query, key)
        attention = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(attention)

        value = x.view(bs, N, -1)
        out = torch.bmm(attention, value)
        out = out.view(bs, N, C, H, W)

        x = self.gamma * out + x
        x = x.view(bs, -1, H, W)
        return x

class CSAM(nn.Module):
    def __init__(self):
        super(CSAM, self).__init__()

        self.conv = nn.Conv3d(1, 1, 3, 1, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))
        # self.softmax  = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        # out = x.unsqueeze(1)
        # out = self.sigmoid(self.conv(out))

        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key) # CxC
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)

        out = self.gamma * out
        out = out.view(m_batchsize, C, height, width)
        x = x * out + x
        return x

class HAN(nn.Module):
    def __init__(self, in_c=3, out_c=3, scale=4, n_feats=128, n_rg=10, n_rcab=20, act=nn.ReLU(True), global_res=False):
        super(HAN, self).__init__()
        self.global_res = global_res

        self.head = conv_(in_c, n_feats)

        self.body = nn.ModuleList()
        for _ in range(n_rg):
            self.body.append(
                ResGroup(n_rcab=n_rcab, n_feats=n_feats, kernel_size=3, stride=1, bias=True, act=act)
            )

        self.csam = CSAM()
        self.lam = LAM()
        self.last_conv = conv_((n_rg+1) * n_feats, n_feats)

        self.tail = nn.Sequential(*[
            UpsampleBlock(scale=scale, n_feats=n_feats, kernel_size=3, stride=1, bias=True, bn=False, act=act),
            conv_(n_feats, out_c)]
          )

    def forward(self, x):
        if self.global_res:
            x0 = nn.Upsample(scale_factor=self.scale, mode='bicubic')(x)
        x = self.head(x)

        res = x
        res1 = []
        for i in range(len(self.body)):
            res = self.body[i](res)
            res1.append(res.unsqueeze(1))

        res1 = torch.cat(res1, dim=1)
        la = self.lam(res1)
        csa = self.csam(res)

        res = torch.cat([la, csa], dim=1)
        res = self.last_conv(res)

        x = x + res
        x = self.tail(x)
        if self.global_res:
            x = x + x0
        return x

if __name__ == "__main__":
    import numpy as np
    import torch
    import torchsummary

    model = HAN(in_c=3, out_c=3, scale=8, n_feats=128, n_rg=10, n_rcab=20, global_res=False)
    print(torchsummary.summary(model, (3, 24, 24), device='cpu'))

    x = np.random.uniform(0, 1, [2, 3, 24, 24]).astype(np.float32)
    x = torch.tensor(x)

    # loss = nn.L1Loss()
    # Adam = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.99, 0.999))
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        y = model(x)
    print(prof)
    print(y.shape)