import torch

from amz_cuda_ext.neuron.lif.cuda_version import LIF

if __name__ == '__main__':
    torch.manual_seed(2024)
    # in_x = torch.rand(3, requires_grad=True).cuda()
    # gt = torch.tensor([0.1, 0.0, 0.0], requires_grad=True).cuda()
    in_x = torch.rand((4, 4), requires_grad=True).cuda()
    gt = torch.rand((4, 4), requires_grad=True).cuda()
    print('x', in_x)
    out_y = LIF()(in_x)
    in_x.retain_grad()
    out_y.retain_grad()
    print('out_y', out_y)
    loss = torch.sum((out_y - gt) ** 2)
    print('loss', loss)
    loss.backward()
    print('out_y.grad', out_y.grad)
    print('x.grad', in_x.grad)

    # x.grad tensor([0.5413, 1.8057, 0.0000])

"""
x tensor([[0.5317, 0.8313, 0.9718, 0.1193],
        [0.1669, 0.3495, 0.2150, 0.6201],
        [0.4849, 0.7492, 0.1521, 0.5625],
        [0.1735, 0.7046, 0.9265, 0.4061]], device='cuda:0', grad_fn=<ToCopyBackward0>)
out_y tensor([[0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.]], device='cuda:0', grad_fn=<MulBackward0>)
loss tensor(5.8878, device='cuda:0', grad_fn=<SumBackward0>)
out_y.grad tensor([[-0.6090, -1.8146, -1.5275, -1.2202],
        [-1.7828, -1.8179, -1.3303, -1.3573],
        [-0.4537,  0.9939, -0.4301, -1.5976],
        [-0.1567, -0.1411,  1.4056, -0.1383]], device='cuda:0')
x.grad tensor([[-0.6499, -1.6941, -1.5088, -0.7929],
        [-0.9093, -1.2018, -0.8776, -1.3133],
        [-0.3390,  0.9118,  0.2158, -1.4527],
        [-0.0822, -0.0994,  1.1557, -0.1186]], device='cuda:0')
"""