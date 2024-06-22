from copy import deepcopy

import torch

from neurons.lif_neuron import LIF_TBN, LIF_BTN, LIF_TBN_Torch, LIF_BTN_Torch


def test_lif_neuron():
    T = 512
    B = 512
    N = 512
    torch.manual_seed(2024)
    x = torch.rand((T, B, N))
    gt = torch.rand((T, B, N)).cuda()

    print('================== TBN ==================')
    lif_tbn_cuda = LIF_TBN()
    lif_tbn_torch = LIF_TBN_Torch()

    x_cuda = deepcopy(x).cuda()
    x_torch = deepcopy(x).cuda()
    x_cuda.requires_grad_()
    x_torch.requires_grad_()
    y_cuda = lif_tbn_cuda(x_cuda)
    y_torch = lif_tbn_torch(x_torch)
    x_cuda.retain_grad()
    x_torch.retain_grad()
    torch.mean((y_cuda - gt) ** 2).backward()
    torch.mean((y_torch - gt) ** 2).backward()

    with torch.no_grad():
        print(f"[INFO] Mean error y      : {torch.mean(x_cuda - x_torch).cpu().numpy()}")
        print(f"[INFO] Max  error y      : {torch.max(x_cuda - x_torch).cpu().numpy()}")
        print(f"[INFO] Min  error y      : {torch.min(x_cuda - x_torch).cpu().numpy()}")
        print(f"[INFO] Mean error y.grade: {torch.mean(x_cuda.grad - x_torch.grad).cpu().numpy()}")
        print(f"[INFO] Max  error y.grade: {torch.max(x_cuda.grad - x_torch.grad).cpu().numpy()}")
        print(f"[INFO] Min  error y.grade: {torch.min(x_cuda.grad - x_torch.grad).cpu().numpy()}")

    x = x.transpose(0, 1).contiguous()
    gt = gt.transpose(0, 1).contiguous()
    print('================== BTN ==================')
    lif_btn_cuda = LIF_BTN()
    lif_btn_torch = LIF_BTN_Torch()

    x_cuda = deepcopy(x).cuda()
    x_torch = deepcopy(x).cuda()
    x_cuda.requires_grad_()
    x_torch.requires_grad_()
    y_cuda = lif_btn_cuda(x_cuda)
    y_torch = lif_btn_torch(x_torch)
    x_cuda.retain_grad()
    x_torch.retain_grad()
    torch.mean((y_cuda - gt) ** 2).backward()
    torch.mean((y_torch - gt) ** 2).backward()

    with torch.no_grad():
        print(f"[INFO] Mean error y      : {torch.mean(x_cuda - x_torch).cpu().numpy()}")
        print(f"[INFO] Max  error y      : {torch.max(x_cuda - x_torch).cpu().numpy()}")
        print(f"[INFO] Min  error y      : {torch.min(x_cuda - x_torch).cpu().numpy()}")
        print(f"[INFO] Mean error y.grade: {torch.mean(x_cuda.grad - x_torch.grad).cpu().numpy()}")
        print(f"[INFO] Max  error y.grade: {torch.max(x_cuda.grad - x_torch.grad).cpu().numpy()}")
        print(f"[INFO] Min  error y.grade: {torch.min(x_cuda.grad - x_torch.grad).cpu().numpy()}")
