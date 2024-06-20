import torch
from torch import nn
from . import lif_forward, lif_backward


class LIFCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_seq, tau, v_th, alpha):
        m_seq, y_seq = lif_forward(x_seq, tau, v_th, alpha)
        if x_seq.requires_grad:
            ctx.save_for_backward(m_seq, y_seq)
            ctx.tau = tau
            ctx.v_th = v_th
            ctx.alpha = alpha
        return y_seq

    @staticmethod
    def backward(ctx, grad_y):
        m = ctx.saved_tensors[0]
        y = ctx.saved_tensors[1]
        grad_x, = lif_backward(grad_y, m, y, ctx.tau, ctx.v_th, ctx.alpha)
        return grad_x, None, None, None


class LIF(nn.Module):
    def __init__(self, thresh=1.0, tau=0.5, alpha=1.0):
        super(LIF, self).__init__()
        self.lif_multi_step = LIFCuda.apply
        self.v_th = thresh
        self.tau = tau
        self.alpha = alpha

    def forward(self, x):
        y = self.lif_multi_step(x, self.tau, self.v_th, self.alpha) * 1.0
        return y

