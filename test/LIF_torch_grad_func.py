import torch
from torch import nn


class LIFFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_seq, tau, v_th, alpha):
        m_seq = []
        y_seq = []
        v = 0.
        for t in range(x_seq.shape[0]):
            m = v * tau + x_seq[t, ...]
            m_seq.append(m)
            y = ((m - v_th) > 0).float()
            y_seq.append(y)
            v = m * (1 - y)

        m_seq = torch.stack(m_seq)
        y_seq = torch.stack(y_seq)
        if x_seq.requires_grad:
            ctx.save_for_backward(m_seq, y_seq)
            ctx.tau = tau
            ctx.v_th = v_th
            ctx.alpha = alpha
        return y_seq

    @staticmethod
    def backward(ctx, grad_y):
        m = ctx.saved_tensors[0]
        y = ctx.saved_tensors[1]
        grad_v = 0.
        grad_m_seq = []
        for t in range(y.shape[0]).__reversed__():
            grad_sg = (ctx.alpha - (m[t, ...] - ctx.v_th).abs()).clamp(min=0) * (1 / ctx.alpha) ** 2
            grad_m = grad_y[t, ...] * grad_sg + grad_v * (1. - m[t, ...] * grad_sg - y[t, ...])
            grad_m_seq.append(grad_m)
            grad_v = grad_m * ctx.tau

        grad_m_seq.reverse()
        grad_m_seq = torch.stack(grad_m_seq)
        grad_x = grad_m_seq
        return grad_x, None, None, None


class LIF(nn.Module):
    def __init__(self, thresh=1.0, tau=0.5, alpha=1.0):
        super(LIF, self).__init__()
        self.lif_multi_step = LIFFunc.apply
        self.v_th = thresh
        self.tau = tau
        self.alpha = alpha

    def forward(self, x):
        y = self.lif_multi_step(x, self.tau, self.v_th, self.alpha) * 1.0
        return y


if __name__ == '__main__':
    torch.manual_seed(2024)
    in_x = torch.rand((4, 4), requires_grad=True)
    gt = torch.rand((4, 4), requires_grad=True)
    print('x', in_x)
    out_y = LIF()(in_x)
    out_y.retain_grad()
    print('out_y', out_y)
    loss = torch.sum((out_y - gt) ** 2)
    print('loss', loss)
    loss.backward()
    print('out_y.grad', out_y.grad)
    print('x.grad', in_x.grad)

"""
x tensor([[0.5317, 0.8313, 0.9718, 0.1193],
        [0.1669, 0.3495, 0.2150, 0.6201],
        [0.4849, 0.7492, 0.1521, 0.5625],
        [0.1735, 0.7046, 0.9265, 0.4061]], requires_grad=True)
out_y tensor([[0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.]], grad_fn=<MulBackward0>)
loss tensor(5.8878, grad_fn=<SumBackward0>)
out_y.grad tensor([[-0.6090, -1.8146, -1.5275, -1.2202],
        [-1.7828, -1.8179, -1.3303, -1.3573],
        [-0.4537,  0.9939, -0.4301, -1.5976],
        [-0.1567, -0.1411,  1.4056, -0.1383]])
x.grad tensor([[-0.6499, -1.6941, -1.5088, -0.7929],
        [-0.9093, -1.2018, -0.8776, -1.3133],
        [-0.3390,  0.9118,  0.2158, -1.4527],
        [-0.0822, -0.0994,  1.1557, -0.1186]])
"""