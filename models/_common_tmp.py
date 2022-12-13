from turtle import forward
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectSlot(nn.Module):
    stride = 64
    export = False

    def __init__(self, nc=2, length=(80, 180), ch=None):
        super().__init__()
        self.nc = nc
        self.no = nc+7 # number of outputs
        self.na = len(length)
        tmp = torch.tensor(length).float().view(1, -1, 1, 1, 1)
        self.register_buffer('length',tmp)
        self.m = nn.Conv2d(ch, self.na*self.no, 1)

    def forward(self, x):
        self.training |= self.export
        # only 1 layer 64x
        x = self.m(x)
        bs, _, ny, nx = x.shape
        x = x.view(bs, self.na, self.no, ny, nx).permute(0,1,3,4,2).contiguous()

        if not self.training:
            self.grid = self._make_grid(nx, ny).to(x.device)

            y = x.sigmoid()
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid) * self.stride
            y[..., 2:6] = y[..., 2:6] * 2. - 1.0
            y[..., 6] = (y[..., 6] * 2) ** 2 * self.length
            z = y.view(bs, -1, self.no)
        
        return x if self.training else (z, x)
        
    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()