import torch

# from amz_ext.test_neurons.test_if_neuron import test_if_all
# from amz_ext.test_neurons.test_lif_neuron import test_lif_all
# from amz_ext.test_neurons.test_if_neuron import test_if_speed, test_if_error
# from amz_ext.test_neurons.test_lif_neuron import test_lif_speed, test_lif_error
from test_neurons.test_if_neuron import test_if_speed as test_if_speed_v1
from test_neurons.test_if_neuron_v2 import test_if_speed as test_if_speed_v2

if __name__ == '__main__':
    test_if_speed_v1(torch.float32, n_iter=500)

    test_if_speed_v1(torch.float16, n_iter=500)

    test_if_speed_v2(torch.float32, n_iter=500)

    test_if_speed_v2(torch.float16, n_iter=500)
