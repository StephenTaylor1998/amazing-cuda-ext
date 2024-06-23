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
    extension = load(
        'cuda_extension',
        [
            os.path.join(src_dir, 'operator_if.cpp'),
            os.path.join(src_dir, 'kernel_if.cu'),
        ],
        build_directory=src_dir,
        verbose=False,
        # extra_cflags={'cxx': ['-O2'], 'nvcc': ['-O3']},
    )

    if_tbn_forward = extension.if_tbn_forward
    if_btn_forward = extension.if_btn_forward
    if_tbn_backward = extension.if_tbn_backward
    if_btn_backward = extension.if_btn_backward

except ModuleNotFoundError:
    warnings.warn("[INFO] Unable to load cuda extensions.")
    if_tbn_forward = None
    if_btn_forward = None
    if_tbn_backward = None
    if_btn_backward = None


class if_tbn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        x_seq, v_th, alpha = args
        m_seq, y_seq = if_tbn_forward(x_seq, v_th)
        if x_seq.requires_grad:
            ctx.save_for_backward(m_seq, y_seq)
            ctx.v_th = v_th
            ctx.alpha = alpha
        return y_seq

    @staticmethod
    def backward(ctx, *args):
        grad_y, = args
        m = ctx.saved_tensors[0]
        y = ctx.saved_tensors[1]
        grad_x, = if_tbn_backward(grad_y, m, y, ctx.v_th, ctx.alpha)
        return grad_x, None, None


class if_btn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        x_seq, v_th, alpha = args
        m_seq, y_seq = if_btn_forward(x_seq, v_th)
        if x_seq.requires_grad:
            ctx.save_for_backward(m_seq, y_seq)
            ctx.v_th = v_th
            ctx.alpha = alpha
        return y_seq

    @staticmethod
    def backward(ctx, *args):
        grad_y, = args
        m = ctx.saved_tensors[0]
        y = ctx.saved_tensors[1]
        grad_x, = if_btn_backward(grad_y, m, y, ctx.v_th, ctx.alpha)
        return grad_x, None, None


class IF(nn.Module):
    def __init__(self, thresh=1.0, alpha=1.0, dim_t=0):
        super(IF, self).__init__()
        assert dim_t in [0, 1], \
            '[amz_cuda_ext] The supported time dimensions are 0 and 1.'
        if dim_t == 0:
            self.if_fun = if_tbn.apply
        else:
            self.if_fun = if_btn.apply
        self.v_th = thresh
        self.alpha = alpha

    def forward(self, x):
        y = self.if_fun(x, self.v_th, self.alpha)
        return y
