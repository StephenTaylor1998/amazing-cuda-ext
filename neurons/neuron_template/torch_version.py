import torch
from torch import nn


class zif_torch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        x, alpha = args
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return (x >= 0).float()

    @staticmethod
    def backward(ctx, *args):
        grad, = args
        factor = ctx.alpha - ctx.saved_tensors[0].abs()
        grad *= factor.clamp(min=0) * (1 / ctx.alpha) ** 2
        return grad, None


class IF_TBN_Torch(nn.Module):
    def __init__(self, thresh=1.0, alpha=1.0):
        super(IF_TBN_Torch, self).__init__()
        self.heaviside = zif_torch.apply
        self.v_th = thresh
        self.gamma = alpha

    def forward(self, x):
        mem_v = []
        mem = 0.
        for t in range(x.shape[0]):
            mem = mem + x[t, ...]
            spike = self.heaviside(mem - self.v_th, self.gamma) * 1.0
            mem = mem * (1. - spike)
            mem_v.append(spike)
        return torch.stack(mem_v)
