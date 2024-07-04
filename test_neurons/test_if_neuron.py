import warnings
from copy import deepcopy

import torch
from tqdm import tqdm

from neurons import IF, IF_TBN_Torch, IF_BTN_Torch


def test_if_all():
    test_if_error(torch.float32)
    test_if_error(torch.float16)

    test_if_speed(torch.float32)
    test_if_speed(torch.float16)


def test_if_error(dtype=torch.float32):
    print(f'\n[amz_ext] [IF] [data_type {dtype}]')
    T = 512
    B = 512
    N = 512
    torch.manual_seed(2024)
    x = torch.rand((T, B, N), dtype=dtype)
    gt = torch.rand((T, B, N), dtype=dtype).cuda()

    print('================== TBN ==================')
    lif_tbn_cuda = IF(dim_t=0)
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
    lif_btn_cuda = IF(dim_t=1)
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


def test_if_speed(dtype=torch.float32, n_iter=50):
    print(f'\n[amz_ext] [IF] [data_type {dtype}]')
    print('================== torch ==================')
    # [RTX4090] [TBN] [6600MB] [36s]
    # [RTX4090] [BTN] [6600MB] [59s]
    test_if_torch(dtype, n_iter)
    print('=============== amz_ext ==============')
    # [RTX4090] [TBN] [3018MB] [3s]
    # [RTX4090] [BTN] [3018MB] [3s]
    test_if_cuda(dtype, n_iter)
    print('================= sjtorch =================')
    # [RTX4090] [TBN] [3530MB] [37s]
    # [RTX4090] [BTN] [7624MB] [37s]
    test_if_spikingjelly_torch(dtype, n_iter)
    print('================= sjcupy ==================')
    # [RTX4090] [TBN] [4048MB] [4s]
    # [RTX4090] [BTN] [4306MB] [4s]
    test_if_spikingjelly_cupy(dtype, n_iter)


def test_if_cuda(dtype=torch.float32, n_iter=50, T=4, B=128, N=1024 * 8 * 8):
    torch.manual_seed(2024)

    x = torch.rand((T, B, N), dtype=dtype)
    gt = torch.rand((T, B, N), dtype=dtype).cuda()
    lif_tbn_cuda = IF(dim_t=0)

    x_cuda = deepcopy(x).cuda()
    with torch.no_grad():
        lif_tbn_cuda(x_cuda)
    for _ in tqdm(range(n_iter)):
        x_cuda.requires_grad_()
        y_cuda = lif_tbn_cuda(x_cuda)
        torch.mean((y_cuda - gt) ** 2).backward()

    x = x.transpose(0, 1).contiguous()
    gt = gt.transpose(0, 1).contiguous()
    lif_btn_cuda = IF(dim_t=1)

    x_cuda = deepcopy(x).cuda()
    with torch.no_grad():
        lif_btn_cuda(x_cuda)
    for _ in tqdm(range(n_iter)):
        x_cuda.requires_grad_()
        y_cuda = lif_btn_cuda(x_cuda)
        torch.mean((y_cuda - gt) ** 2).backward()


def test_if_torch(dtype=torch.float32, n_iter=50, T=4, B=128, N=1024 * 8 * 8):
    torch.manual_seed(2024)

    x = torch.rand((T, B, N), dtype=dtype)
    gt = torch.rand((T, B, N), dtype=dtype).cuda()
    lif_tbn_cuda = IF_TBN_Torch()

    x_cuda = deepcopy(x).cuda()
    with torch.no_grad():
        lif_tbn_cuda(x_cuda)
    for _ in tqdm(range(n_iter)):
        x_cuda.requires_grad_()
        y_cuda = lif_tbn_cuda(x_cuda)
        torch.mean((y_cuda - gt) ** 2).backward()

    x = x.transpose(0, 1).contiguous()
    gt = gt.transpose(0, 1).contiguous()
    lif_btn_cuda = IF_BTN_Torch()

    x_cuda = deepcopy(x).cuda()
    with torch.no_grad():
        lif_btn_cuda(x_cuda)
    for _ in tqdm(range(n_iter)):
        x_cuda.requires_grad_()
        y_cuda = lif_btn_cuda(x_cuda)
        torch.mean((y_cuda - gt) ** 2).backward()


def test_if_spikingjelly_torch(dtype=torch.float32, n_iter=50, T=4, B=128, N=1024 * 8 * 8):
    try:
        from spikingjelly.activation_based.neuron import IFNode
        from spikingjelly.activation_based.functional import reset_net
    except ImportWarning:
        warnings.warn("[INFO] Unable to load cuda extensions.")
        return
    torch.manual_seed(2024)

    x = torch.rand((T, B, N), dtype=dtype)
    gt = torch.rand((T, B, N), dtype=dtype).cuda()
    lif_tbn_cuda = IFNode(backend='torch', step_mode='m')

    x_cuda = deepcopy(x).cuda()
    with torch.no_grad():
        lif_tbn_cuda(x_cuda)
    for _ in tqdm(range(n_iter)):
        reset_net(lif_tbn_cuda)
        x_cuda.requires_grad_()
        y_cuda = lif_tbn_cuda(x_cuda)
        torch.mean((y_cuda - gt) ** 2).backward()


def test_if_spikingjelly_cupy(dtype=torch.float32, n_iter=50, T=4, B=128, N=1024 * 8 * 8):
    try:
        from spikingjelly.activation_based.neuron import IFNode
        from spikingjelly.activation_based.functional import reset_net
    except ImportWarning:
        warnings.warn("[INFO] Unable to load cuda extensions.")
        return
    torch.manual_seed(2024)

    x = torch.rand((T, B, N), dtype=dtype)
    gt = torch.rand((T, B, N), dtype=dtype).cuda()
    lif_tbn_cuda = IFNode(backend='cupy', step_mode='m')

    x_cuda = deepcopy(x).cuda()
    with torch.no_grad():
        lif_tbn_cuda(x_cuda)
    for _ in tqdm(range(n_iter)):
        reset_net(lif_tbn_cuda)
        x_cuda.requires_grad_()
        y_cuda = lif_tbn_cuda(x_cuda)
        torch.mean((y_cuda - gt) ** 2).backward()
