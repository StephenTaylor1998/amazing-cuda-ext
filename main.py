import torch

from amz_cuda_ext.test_neuron.lif import run_eval

if __name__ == '__main__':
    torch.manual_seed(2024)
    run_eval()
