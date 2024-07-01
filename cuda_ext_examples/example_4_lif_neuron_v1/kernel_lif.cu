//
// Created by stephen on 24-6-22.
//
#include <torch/extension.h>
#include <vector>

template<typename scalar_t>
__global__ void lif_tbn_forward_kernel(
        int T, int BN, const scalar_t *x, const scalar_t tau,
        const scalar_t v_th, scalar_t *y, scalar_t *m) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    scalar_t v = 0.0;
    if (idx < BN) {
        for (int t = 0; t < T; t++) {
            m[idx + t * BN] = v * tau + x[idx + t * BN];
            y[idx + t * BN] = (m[idx + t * BN] - v_th) >= (scalar_t) 0.0 ? (scalar_t) 1.0 : (scalar_t) 0.0;
            v = m[idx + t * BN] * (1 - y[idx + t * BN]);
        }
    }
}

template<typename scalar_t>
__global__ void
lif_btn_forward_kernel(
        int numel, int T, int N, const scalar_t *x, const scalar_t tau,
        const scalar_t v_th, scalar_t *y, scalar_t *m) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    scalar_t v = 0.0;
    if ((idx % (N * T)) < N) {
        for (int t = 0; t < T; t++) {
            m[idx + t * N] = v * tau + x[idx + t * N];
            y[idx + t * N] = (m[idx + t * N] - v_th) >= (scalar_t) 0.0 ? (scalar_t) 1.0 : (scalar_t) 0.0;
            v = m[idx + t * N] * (1 - y[idx + t * N]);
        }
    }
}

template<typename scalar_t>
__global__ void lif_tbn_backward_kernel(
        int T, int BN, const scalar_t *grad_y, const scalar_t *y, const scalar_t *m,
        const scalar_t tau, const scalar_t v_th, const scalar_t alpha, scalar_t *grad_x) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    scalar_t grad_v = 0.0;
    if (idx < BN) {
        for (int t = T - 1; t >= 0; t--) {
            scalar_t grad_sg_1 = (alpha - abs(m[idx + t * BN] - v_th));
            scalar_t clamp = grad_sg_1 < (scalar_t) 0.0 ? (scalar_t) 0.0 : grad_sg_1;
            scalar_t grad_sg = clamp * (1. / alpha) * (1. / alpha);
            scalar_t grad_m =
                    grad_y[idx + t * BN] * grad_sg + grad_v * (1. - m[idx + t * BN] * grad_sg - y[idx + t * BN]);
            grad_x[idx + t * BN] = grad_m;
            grad_v = grad_m * tau;
        }
    }
}

template<typename scalar_t>
__global__ void lif_btn_backward_kernel(
        int numel, int T, int N, const scalar_t *grad_y, const scalar_t *y, const scalar_t *m,
        const scalar_t tau, const scalar_t v_th, const scalar_t alpha, scalar_t *grad_x) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    scalar_t grad_v = 0.0;
    if ((idx % (N * T)) < N) {
        for (int t = T - 1; t >= 0; t--) {
            scalar_t grad_sg_1 = (alpha - abs(m[idx + t * N] - v_th));
            scalar_t clamp = grad_sg_1 < (scalar_t) 0.0f ? (scalar_t) 0.0 : grad_sg_1;
            scalar_t grad_sg = clamp * (1. / alpha) * (1. / alpha);
            scalar_t grad_m = grad_y[idx + t * N] * grad_sg + grad_v * (1. - m[idx + t * N] * grad_sg - y[idx + t * N]);
            grad_x[idx + t * N] = grad_m;
            grad_v = grad_m * tau;
        }
    }
}

std::vector<at::Tensor> lif_tbn_forward(const at::Tensor x, float tau, float v_th) {
    TORCH_CHECK(x.dtype() == at::kHalf || x.dtype() == at::kFloat);
    TORCH_INTERNAL_ASSERT(x.is_contiguous());
    TORCH_INTERNAL_ASSERT(x.device().type() == at::DeviceType::CUDA);

    at::Tensor y = torch::empty(x.sizes(), x.options());
    at::Tensor m = torch::empty(x.sizes(), x.options());

    int numel = x.numel();
    int T = x.size(0);
    int BN = numel / T;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "lif_tbn_forward_kernel", ([&] {
        lif_tbn_forward_kernel<scalar_t><<<numel / 256 + 1, 256>>>(
                T, BN, x.data_ptr<scalar_t>(), tau, v_th, y.data_ptr<scalar_t>(), m.data_ptr<scalar_t>()
        );
    }));

    return {m, y};
}

std::vector<at::Tensor> lif_btn_forward(const at::Tensor x, float tau, float v_th) {
    TORCH_CHECK(x.dtype() == at::kHalf || x.dtype() == at::kFloat);
    TORCH_INTERNAL_ASSERT(x.is_contiguous());
    TORCH_INTERNAL_ASSERT(x.device().type() == at::DeviceType::CUDA);

    at::Tensor y = torch::empty(x.sizes(), x.options());
    at::Tensor m = torch::empty(x.sizes(), x.options());

    int numel = x.numel();
    int B = x.size(0);
    int T = x.size(1);
    int N = numel / (B * T);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "lif_btn_forward_kernel", ([&] {
        lif_btn_forward_kernel<scalar_t><<<numel / 256 + 1, 256>>>(
                numel, T, N, x.data_ptr<scalar_t>(), tau, v_th, y.data_ptr<scalar_t>(), m.data_ptr<scalar_t>()
        );
    }));
    return {m, y};
}

std::vector<at::Tensor> lif_tbn_backward(const at::Tensor grad_y, const at::Tensor m, const at::Tensor y,
                                         float tau, float v_th, float alpha) {
    TORCH_CHECK(grad_y.dtype() == at::kHalf || grad_y.dtype() == at::kFloat);
    TORCH_INTERNAL_ASSERT(grad_y.is_contiguous());
    TORCH_INTERNAL_ASSERT(grad_y.device().type() == at::DeviceType::CUDA);

    TORCH_CHECK(m.dtype() == at::kHalf || m.dtype() == at::kFloat);
    TORCH_INTERNAL_ASSERT(m.is_contiguous());
    TORCH_INTERNAL_ASSERT(m.device().type() == at::DeviceType::CUDA);

    TORCH_CHECK(y.dtype() == at::kHalf || y.dtype() == at::kFloat);
    TORCH_INTERNAL_ASSERT(y.is_contiguous());
    TORCH_INTERNAL_ASSERT(y.device().type() == at::DeviceType::CUDA);

    at::Tensor grad_x = torch::empty(grad_y.sizes(), grad_y.options());

    int numel = grad_y.numel();
    int T = grad_y.size(0);
    int BN = numel / T;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_y.scalar_type(), "lif_tbn_backward_kernel", ([&] {
        lif_tbn_backward_kernel<scalar_t><<<numel / 256 + 1, 256>>>(
                T, BN, grad_y.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(),
                m.data_ptr<scalar_t>(), tau, v_th, alpha, grad_x.data_ptr<scalar_t>()
        );
    }));

    return {grad_x};
}

std::vector<at::Tensor> lif_btn_backward(const at::Tensor grad_y, const at::Tensor m, const at::Tensor y,
                                         float tau, float v_th, float alpha) {
    TORCH_CHECK(grad_y.dtype() == at::kHalf || grad_y.dtype() == at::kFloat);
    TORCH_INTERNAL_ASSERT(grad_y.is_contiguous());
    TORCH_INTERNAL_ASSERT(grad_y.device().type() == at::DeviceType::CUDA);

    TORCH_CHECK(m.dtype() == at::kHalf || m.dtype() == at::kFloat);
    TORCH_INTERNAL_ASSERT(m.is_contiguous());
    TORCH_INTERNAL_ASSERT(m.device().type() == at::DeviceType::CUDA);

    TORCH_CHECK(y.dtype() == at::kHalf || y.dtype() == at::kFloat);
    TORCH_INTERNAL_ASSERT(y.is_contiguous());
    TORCH_INTERNAL_ASSERT(y.device().type() == at::DeviceType::CUDA);

    at::Tensor grad_x = torch::empty(grad_y.sizes(), grad_y.options());

    int numel = grad_y.numel();
    int B = grad_y.size(0);
    int T = grad_y.size(1);
    int N = numel / (B * T);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_y.scalar_type(), "lif_btn_backward_kernel", ([&] {
        lif_btn_backward_kernel<scalar_t><<<numel / 256 + 1, 256>>>(
                numel, T, N, grad_y.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(),
                m.data_ptr<scalar_t>(), tau, v_th, alpha, grad_x.data_ptr<scalar_t>()
        );
    }));
    return {grad_x};
}

