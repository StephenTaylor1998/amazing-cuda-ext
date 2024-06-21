from copy import deepcopy

import torch

from neuron.lif import LIF_TBN, LIF_BTN, LIF_TBN_Torch, LIF_BTN_Torch


def run_eval():
    torch.manual_seed(2024)
    T = 16
    B = 4
    N = 128
    x = torch.rand((T, B, N), requires_grad=True)
    gt = torch.rand((T, B, N)).cuda()

    # ================== TBN ==================
    print('================== TBN ==================')
    lif_tbn_cuda = LIF_TBN()
    lif_tbn_torch = LIF_TBN_Torch()
    x_cuda = deepcopy(x).cuda()
    x_torch = deepcopy(x).cuda()
    y_cuda = lif_tbn_cuda(x_cuda)
    y_torch = lif_tbn_torch(x_torch)
    x_cuda.retain_grad()
    x_torch.retain_grad()
    torch.mean((y_cuda - gt) ** 2).backward()
    torch.mean((y_torch - gt) ** 2).backward()
    print(f"[INFO] Mean error: {torch.mean(x_cuda.grad - x_torch.grad).cpu().numpy()}")
    print(f"[INFO] Max  error: {torch.max(x_cuda.grad - x_torch.grad).cpu().numpy()}")
    print(f"[INFO] Min  error: {torch.min(x_cuda.grad - x_torch.grad).cpu().numpy()}")

    # ================== BTN ==================
    print('================== BTN ==================')
    lif_btn_cuda = LIF_BTN()
    lif_btn_torch = LIF_BTN_Torch()

    x_cuda = deepcopy(x).cuda()
    x_torch = deepcopy(x).cuda()
    y_cuda = lif_btn_cuda(x_cuda)
    y_torch = lif_btn_torch(x_torch)
    x_cuda.retain_grad()
    x_torch.retain_grad()
    torch.mean((y_cuda - gt) ** 2).backward()
    torch.mean((y_torch - gt) ** 2).backward()
    print(f"[INFO] Mean error: {torch.mean(x_cuda.grad - x_torch.grad).cpu().numpy()}")
    print(f"[INFO] Max  error: {torch.max(x_cuda.grad - x_torch.grad).cpu().numpy()}")
    print(f"[INFO] Min  error: {torch.min(x_cuda.grad - x_torch.grad).cpu().numpy()}")
