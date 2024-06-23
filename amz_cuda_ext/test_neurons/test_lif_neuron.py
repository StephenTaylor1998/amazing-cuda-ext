import warnings
from copy import deepcopy

import torch
from tqdm import tqdm

from neurons.lif_neuron import LIF, LIF_TBN_Torch, LIF_BTN_Torch


def test_lif_all():
    test_lif_error(torch.float32)
    test_lif_error(torch.float16)

    test_lif_speed(torch.float32)
    test_lif_speed(torch.float16)


def test_lif_error(dtype=torch.float32):
    print(f'\n[amz_cuda_ext] [LIF] [data_type {dtype}]')
    T = 512
    B = 512
    N = 512
    torch.manual_seed(2024)
    x = torch.rand((T, B, N), dtype=dtype)
    gt = torch.rand((T, B, N), dtype=dtype).cuda()

    print('================== TBN ==================')
    lif_tbn_cuda = LIF(dim_t=0)
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
    lif_btn_cuda = LIF(dim_t=1)
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


def test_lif_speed(dtype=torch.float32, n_iter=50):
    print(f'\n[amz_cuda_ext] [LIF] [data_type {dtype}]')

    def test_lif_cuda(dtype=torch.float32, n_iter=50):
        T = 4
        B = 128
        N = 4096
        torch.manual_seed(2024)

        x = torch.rand((T, B, N), dtype=dtype)
        gt = torch.rand((T, B, N), dtype=dtype).cuda()
        lif_tbn_cuda = LIF(dim_t=0)

        for _ in tqdm(range(n_iter)):
            x_cuda = deepcopy(x).cuda()
            x_cuda.requires_grad_()
            y_cuda = lif_tbn_cuda(x_cuda)
            x_cuda.retain_grad()
            torch.mean((y_cuda - gt) ** 2).backward()

        x = x.transpose(0, 1).contiguous()
        gt = gt.transpose(0, 1).contiguous()
        lif_btn_cuda = LIF(dim_t=1)

        for _ in tqdm(range(n_iter)):
            x_cuda = deepcopy(x).cuda()
            x_cuda.requires_grad_()
            y_cuda = lif_btn_cuda(x_cuda)
            x_cuda.retain_grad()
            torch.mean((y_cuda - gt) ** 2).backward()

    def test_lif_torch(dtype=torch.float32, n_iter=50):
        T = 4
        B = 128
        N = 4096
        torch.manual_seed(2024)

        x = torch.rand((T, B, N), dtype=dtype)
        gt = torch.rand((T, B, N), dtype=dtype).cuda()
        lif_tbn_cuda = LIF_TBN_Torch()

        for _ in tqdm(range(n_iter)):
            x_cuda = deepcopy(x).cuda()
            x_cuda.requires_grad_()
            y_cuda = lif_tbn_cuda(x_cuda)
            x_cuda.retain_grad()
            torch.mean((y_cuda - gt) ** 2).backward()

        x = x.transpose(0, 1).contiguous()
        gt = gt.transpose(0, 1).contiguous()
        lif_btn_cuda = LIF_BTN_Torch()

        for _ in tqdm(range(n_iter)):
            x_cuda = deepcopy(x).cuda()
            x_cuda.requires_grad_()
            y_cuda = lif_btn_cuda(x_cuda)
            x_cuda.retain_grad()
            torch.mean((y_cuda - gt) ** 2).backward()

    def test_lif_spikingjelly_torch(dtype=torch.float32, n_iter=50):
        try:
            from spikingjelly.activation_based.neuron import LIFNode
            from spikingjelly.activation_based.functional import reset_net
        except ImportWarning:
            warnings.warn("[INFO] Unable to load cuda extensions.")
            return
        T = 4
        B = 128
        N = 4096
        torch.manual_seed(2024)

        x = torch.rand((T, B, N), dtype=dtype)
        gt = torch.rand((T, B, N), dtype=dtype).cuda()
        lif_tbn_cuda = LIFNode(backend='torch', step_mode='m')

        for _ in tqdm(range(n_iter)):
            reset_net(lif_tbn_cuda)
            x_cuda = deepcopy(x).cuda()
            x_cuda.requires_grad_()
            y_cuda = lif_tbn_cuda(x_cuda)
            x_cuda.retain_grad()
            torch.mean((y_cuda - gt) ** 2).backward()

    def test_lif_spikingjelly_cupy(dtype=torch.float32, n_iter=50):
        try:
            from spikingjelly.activation_based.neuron import LIFNode
            from spikingjelly.activation_based.functional import reset_net
        except ImportWarning:
            warnings.warn("[INFO] Unable to load cuda extensions.")
            return
        T = 4
        B = 128
        N = 4096
        torch.manual_seed(2024)

        x = torch.rand((T, B, N), dtype=dtype)
        gt = torch.rand((T, B, N), dtype=dtype).cuda()
        lif_tbn_cuda = LIFNode(backend='cupy', step_mode='m')

        for _ in tqdm(range(n_iter)):
            reset_net(lif_tbn_cuda)
            x_cuda = deepcopy(x).cuda()
            x_cuda.requires_grad_()
            y_cuda = lif_tbn_cuda(x_cuda)
            x_cuda.retain_grad()
            torch.mean((y_cuda - gt) ** 2).backward()

    print('================== torch ==================')
    # [RTX4090] [TBN] [6600MB] [36s]
    # [RTX4090] [BTN] [6600MB] [59s]
    test_lif_torch(dtype, n_iter)
    print('=============== amz_cuda_ext ==============')
    # [RTX4090] [TBN] [3018MB] [3s]
    # [RTX4090] [BTN] [3018MB] [3s]
    test_lif_cuda(dtype, n_iter)
    print('================= sjtorch =================')
    # [RTX4090] [TBN] [3530MB] [37s]
    # [RTX4090] [BTN] [7624MB] [37s]
    test_lif_spikingjelly_torch(dtype, n_iter)
    print('================= sjcupy ==================')
    # [RTX4090] [TBN] [4048MB] [4s]
    # [RTX4090] [BTN] [4306MB] [4s]
    test_lif_spikingjelly_cupy(dtype, n_iter)
