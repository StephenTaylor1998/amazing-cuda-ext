from copy import deepcopy

import torch

from neurons.if_neuron import IF_TBN, IF_BTN, IF_TBN_Torch, IF_BTN_Torch


def test_if_neuron():
    T = 512
    B = 512
    N = 512
    torch.manual_seed(2024)
    x = torch.rand((T, B, N))
    gt = torch.rand((T, B, N)).cuda()

    print('================== TBN ==================')
    lif_tbn_cuda = IF_TBN()
    lif_tbn_torch = IF_TBN_Torch()

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
    lif_btn_cuda = IF_BTN()
    lif_btn_torch = IF_BTN_Torch()

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
