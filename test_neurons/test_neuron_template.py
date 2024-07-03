from copy import deepcopy

import torch

from neurons.neuron_template import IF, IF_TBN_Torch


def test_neuron_template():
    test_template(torch.float32)
    test_template(torch.float16)


def test_template(dtype=torch.float32):
    print(f'\n[amz_cuda_ext] [NT] [data_type {dtype}]')
    T = 512
    B = 512
    N = 512
    torch.manual_seed(2024)
    x = torch.rand((T, B, N), dtype=dtype)
    gt = torch.rand((T, B, N), dtype=dtype).cuda()

    print('================== TBN ==================')
    lif_tbn_cuda = IF()
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
