import torch

# from amz_ext.test_neurons.test_if_neuron import test_if_all
# from amz_ext.test_neurons.test_lif_neuron import test_lif_all
# from amz_ext.test_neurons.test_if_neuron import test_if_speed, test_if_error
# from amz_ext.test_neurons.test_lif_neuron import test_lif_speed, test_lif_error
# from test_neurons.test_if_neuron import test_if_speed as test_if_speed_v1
from test_neurons.test_if_neuron import *
from test_neurons.test_lif_neuron import *

if __name__ == '__main__':

    test_if_torch(
        dtype=torch.float32, n_iter=500, T=4, B=128, N=65536
    )
    test_if_spikingjelly_torch(
        dtype=torch.float32, n_iter=500, T=4, B=128, N=65536
    )
    test_if_spikingjelly_cupy(
        dtype=torch.float32, n_iter=500, T=4, B=128, N=65536
    )
    test_if_cuda(
        dtype=torch.float32, n_iter=500, T=4, B=128, N=65536
    )

    # test_lif_torch(
    #     dtype=torch.float32, n_iter=500, T=4, B=128, N=65536
    # )
    # test_lif_spikingjelly_torch(
    #     dtype=torch.float32, n_iter=500, T=4, B=128, N=65536
    # )
    # test_lif_spikingjelly_cupy(
    #     dtype=torch.float32, n_iter=500, T=4, B=128, N=65536
    # )
    # test_lif_cuda(
    #     dtype=torch.float32, n_iter=500, T=4, B=128, N=65536
    # )