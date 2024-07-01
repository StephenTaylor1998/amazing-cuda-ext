import os
import warnings

import torch
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
        ],
        build_directory=src_dir,
        verbose=False,
        extra_cflags=['-O3'],
        extra_cuda_cflags=["-O3", "-Xfatbin", "-compress-all"],
    )

    cuda_add = extension.add
    cuda_sub = extension.sub

except ModuleNotFoundError:
    warnings.warn("[INFO] Unable to load cuda extensions.")
    cuda_add = None
    cuda_sub = None

