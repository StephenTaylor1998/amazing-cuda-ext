import torch

# from amz_cuda_ext.test_neurons.test_if_neuron import test_if_all
# from amz_cuda_ext.test_neurons.test_lif_neuron import test_lif_all
from amz_cuda_ext.test_neurons.test_if_neuron import test_if_speed, test_if_error
from amz_cuda_ext.test_neurons.test_lif_neuron import test_lif_speed, test_lif_error

if __name__ == '__main__':
    # test_if_all()
    # test_lif_all()

    test_if_error(torch.float32)
    test_if_speed(torch.float32, n_iter=5000)

    test_if_error(torch.float16)
    test_if_speed(torch.float16, n_iter=5000)

    test_lif_error(torch.float32)
    test_lif_speed(torch.float32, n_iter=5000)

    test_lif_error(torch.float16)
    test_lif_speed(torch.float16, n_iter=5000)
