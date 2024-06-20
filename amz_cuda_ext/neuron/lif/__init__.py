import os
import warnings

import torch
from torch.utils.cpp_extension import load

try:
    assert torch.cuda.is_available()
    CUDA_ARCH = torch.cuda.get_device_capability()
    os.environ['TORCH_CUDA_ARCH_LIST'] = f'{CUDA_ARCH[0]}.{CUDA_ARCH[1]}'
    os.environ['MAX_JOBS'] = '4'

    src_dir = os.path.dirname(os.path.realpath(__file__))
    extension = load('cuda_extension', [
        os.path.join(src_dir, 'operator_lif.cpp'),
        os.path.join(src_dir, 'kernel_lif.cu'),
    ], build_directory=src_dir, verbose=True)

    lif_forward = extension.forward_lif
    lif_backward = extension.backward_lif

except ModuleNotFoundError:
    warnings.warn("[INFO] Unable to load cuda extensions.")
    lif_forward = None
    lif_backward = None

__all__ = ['lif_forward', 'lif_backward']
