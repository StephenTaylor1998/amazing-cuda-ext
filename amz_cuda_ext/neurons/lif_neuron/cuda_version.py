import os
import warnings

import torch
from torch import nn
from torch.utils.cpp_extension import load

try:
    assert torch.cuda.is_available()
    CUDA_ARCH = torch.cuda.get_device_capability()
    os.environ['TORCH_CUDA_ARCH_LIST'] = f'{CUDA_ARCH[0]}.{CUDA_ARCH[1]}'
    os.environ['MAX_JOBS'] = '8'

    src_dir = os.path.dirname(os.path.realpath(__file__))
    extension = load('cuda_extension', [
        os.path.join(src_dir, 'operator_lif.cpp'),
        os.path.join(src_dir, 'kernel_lif.cu'),
    ], build_directory=src_dir, verbose=False)

    lif_tbn_forward = extension.lif_tbn_forward
    lif_btn_forward = extension.lif_btn_forward
    lif_tbn_backward = extension.lif_tbn_backward
    lif_btn_backward = extension.lif_btn_backward

except ModuleNotFoundError:
    warnings.warn("[INFO] Unable to load cuda extensions.")
    lif_tbn_forward = None
    lif_btn_forward = None
    lif_tbn_backward = None
    lif_btn_backward = None


class lif_tbn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        x_seq, tau, v_th, alpha = args
        m_seq, y_seq = lif_tbn_forward(x_seq, tau, v_th, alpha)
        if x_seq.requires_grad:
            ctx.save_for_backward(m_seq, y_seq)
            ctx.tau = tau
            ctx.v_th = v_th
            ctx.alpha = alpha
        return y_seq

    @staticmethod
    def backward(ctx, *args):
        grad_y, = args
        m = ctx.saved_tensors[0]
        y = ctx.saved_tensors[1]
        grad_x, = lif_tbn_backward(grad_y, m, y, ctx.tau, ctx.v_th, ctx.alpha)
        return grad_x, None, None, None


class lif_btn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        x_seq, tau, v_th, alpha = args
        m_seq, y_seq = lif_btn_forward(x_seq, tau, v_th, alpha)
        if x_seq.requires_grad:
            ctx.save_for_backward(m_seq, y_seq)
            ctx.tau = tau
            ctx.v_th = v_th
            ctx.alpha = alpha
        return y_seq

    @staticmethod
    def backward(ctx, *args):
        grad_y, = args
        m = ctx.saved_tensors[0]
        y = ctx.saved_tensors[1]
        grad_x, = lif_btn_backward(grad_y, m, y, ctx.tau, ctx.v_th, ctx.alpha)
        return grad_x, None, None, None


class LIF_TBN(nn.Module):
    def __init__(self, thresh=1.0, tau=0.5, alpha=1.0):
        super(LIF_TBN, self).__init__()
        self.lif_tbn = lif_tbn.apply
        self.v_th = thresh
        self.tau = tau
        self.alpha = alpha

    def forward(self, x):
        y = self.lif_tbn(x, self.tau, self.v_th, self.alpha) * 1.0
        return y


class LIF_BTN(nn.Module):
    def __init__(self, thresh=1.0, tau=0.5, alpha=1.0):
        super(LIF_BTN, self).__init__()
        self.lif_btn = lif_btn.apply
        self.v_th = thresh
        self.tau = tau
        self.alpha = alpha

    def forward(self, x):
        y = self.lif_btn(x, self.tau, self.v_th, self.alpha) * 1.0
        return y