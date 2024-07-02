//
// Created by Hangchi Shen on 24-6-23.
//
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <vector>

template<typename scalar_t>
__device__ __forceinline__ void heaviside(scalar_t &m, scalar_t v_th, scalar_t &y) {
    y = ((m - v_th) >= (scalar_t) 0.0) ? (scalar_t) 1.0 : (scalar_t) 0.0;
}

template<typename scalar_t>
__device__ __forceinline__ void reset_v(scalar_t &m, scalar_t &y, scalar_t &v) {
    // v = m[idx + t * BN] * (1 - y[idx + t * BN]);
    v = m * ((scalar_t) 1.0 - y);
    // v = (y >= (scalar_t) 1.0) ? (scalar_t) 0.0 : m;
}

template<typename scalar_t>
__device__ __forceinline__ void clamp_min(scalar_t &x, scalar_t min_val) {
    x = (x > min_val) ? x : min_val;
}

template<typename scalar_t>
__device__ __forceinline__ void clamp_max(scalar_t &x, scalar_t max_val) {
    x = (x < max_val) ? x : max_val;
}

template<typename scalar_t>
__device__ __forceinline__ void surrogate_grad(scalar_t m, scalar_t alpha, scalar_t v_th, scalar_t &grad_sg) {
    scalar_t tmp = (alpha - abs(m - v_th));
    clamp_min(tmp, (scalar_t) 0.0);
    grad_sg = tmp * (1. / alpha) * (1. / alpha);
}

template<typename scalar_t>
__global__ void if_tbn_forward_kernel(
        int T, int BN, const scalar_t *x, const scalar_t v_th, scalar_t *y, scalar_t *m) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    scalar_t v = 0.0;
    if (idx < BN) {
        for (int t = 0; t < T; t++) {
            m[idx + t * BN] = v + x[idx + t * BN];
            heaviside<scalar_t>(m[idx + t * BN], v_th, y[idx + t * BN]);
            reset_v<scalar_t>(m[idx + t * BN], y[idx + t * BN], v);
        }
    }
}

template<typename scalar_t>
__global__ void if_btn_forward_kernel(int numel, int T, int N, const scalar_t *x, const scalar_t v_th,
                                      scalar_t *y, scalar_t *m) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    scalar_t v = 0.0;
    if ((idx % (N * T)) < N) {
        for (int t = 0; t < T; t++) {
            m[idx + t * N] = v + x[idx + t * N];
            y[idx + t * N] = (m[idx + t * N] - v_th) >= (scalar_t) 0.0 ? (scalar_t) 1.0 : (scalar_t) 0.0;
            v = m[idx + t * N] * (1 - y[idx + t * N]);
        }
    }
}

template<typename scalar_t>
__global__ void if_tbn_backward_kernel(int T, int BN, const scalar_t *grad_y, const scalar_t *y, const scalar_t *m,
                                       const scalar_t v_th, const scalar_t alpha, scalar_t *grad_x) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    scalar_t grad_v = 0.0;
    if (idx < BN) {
        for (int t = T - 1; t >= 0; t--) {
            scalar_t grad_sg;
            surrogate_grad<scalar_t>(m[idx + t * BN], alpha, v_th, grad_sg);
            scalar_t grad_m =
                    grad_y[idx + t * BN] * grad_sg + grad_v * (1. - m[idx + t * BN] * grad_sg - y[idx + t * BN]);
            grad_x[idx + t * BN] = grad_m;
            grad_v = grad_m;
        }
    }
}

template<typename scalar_t>
__global__ void
if_btn_backward_kernel(int numel, int T, int N, const scalar_t *grad_y, const scalar_t *y, const scalar_t *m,
                       const scalar_t v_th, const scalar_t alpha, scalar_t *grad_x) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    scalar_t grad_v = 0.0;
    if ((idx % (N * T)) < N) {
        for (int t = T - 1; t >= 0; t--) {
            scalar_t grad_sg_1 = (alpha - abs(m[idx + t * N] - v_th));
            scalar_t clamp = grad_sg_1 < (scalar_t) 0.0 ? (scalar_t) 0.0 : grad_sg_1;
            scalar_t grad_sg = clamp * (1. / alpha) * (1. / alpha);
            scalar_t grad_m = grad_y[idx + t * N] * grad_sg + grad_v * (1. - m[idx + t * N] * grad_sg - y[idx + t * N]);
            grad_x[idx + t * N] = grad_m;
            grad_v = grad_m;
        }
    }
}

std::vector<at::Tensor> if_tbn_forward(const at::Tensor x, float v_th) {
    TORCH_CHECK(x.dtype() == at::kHalf || x.dtype() == at::kFloat);
    TORCH_INTERNAL_ASSERT(x.is_contiguous());
    TORCH_INTERNAL_ASSERT(x.device().type() == at::DeviceType::CUDA);

    at::Tensor y = torch::zeros(x.sizes(), x.options());
    at::Tensor m = torch::zeros(x.sizes(), x.options());

    int numel = x.numel();
    int T = x.size(0);
    int BN = numel / T;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "if_tbn_forward_kernel", ([&] {
        if_tbn_forward_kernel<scalar_t><<<numel / 1024 + 1, 1024>>>(
                T, BN, x.data_ptr<scalar_t>(), v_th, y.data_ptr<scalar_t>(), m.data_ptr<scalar_t>()
        );
    }));

    return {m, y};
}

std::vector<at::Tensor> if_btn_forward(const at::Tensor x, float v_th) {
    TORCH_CHECK(x.dtype() == at::kHalf || x.dtype() == at::kFloat);
    TORCH_INTERNAL_ASSERT(x.is_contiguous());
    TORCH_INTERNAL_ASSERT(x.device().type() == at::DeviceType::CUDA);

    at::Tensor y = torch::zeros(x.sizes(), x.options());
    at::Tensor m = torch::zeros(x.sizes(), x.options());

    int numel = x.numel();
    int B = x.size(0);
    int T = x.size(1);
    int N = numel / (B * T);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "if_btn_forward_kernel", ([&] {
        if_btn_forward_kernel<scalar_t><<<numel / 1024 + 1, 1024>>>(
                numel, T, N, x.data_ptr<scalar_t>(), v_th, y.data_ptr<scalar_t>(), m.data_ptr<scalar_t>()
        );
    }));
    return {m, y};
}

std::vector<at::Tensor> if_tbn_backward(const at::Tensor grad_y, const at::Tensor m, const at::Tensor y,
                                        float v_th, float alpha) {
    TORCH_CHECK(grad_y.dtype() == at::kHalf || grad_y.dtype() == at::kFloat);
    TORCH_INTERNAL_ASSERT(grad_y.is_contiguous());
    TORCH_INTERNAL_ASSERT(grad_y.device().type() == at::DeviceType::CUDA);

    TORCH_CHECK(m.dtype() == at::kHalf || m.dtype() == at::kFloat);
    TORCH_INTERNAL_ASSERT(m.is_contiguous());
    TORCH_INTERNAL_ASSERT(m.device().type() == at::DeviceType::CUDA);

    TORCH_CHECK(y.dtype() == at::kHalf || y.dtype() == at::kFloat);
    TORCH_INTERNAL_ASSERT(y.is_contiguous());
    TORCH_INTERNAL_ASSERT(y.device().type() == at::DeviceType::CUDA);

    at::Tensor grad_x = torch::zeros(grad_y.sizes(), grad_y.options());

    int numel = grad_y.numel();
    int T = grad_y.size(0);
    int BN = numel / T;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_y.scalar_type(), "if_tbn_backward_kernel", ([&] {
        if_tbn_backward_kernel<scalar_t><<<numel / 1024 + 1, 1024>>>(
                T, BN, grad_y.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), m.data_ptr<scalar_t>(), v_th, alpha, grad_x.data_ptr<scalar_t>()
        );
    }));

    return {grad_x};
}

std::vector<at::Tensor> if_btn_backward(const at::Tensor grad_y, const at::Tensor m, const at::Tensor y,
                                        float v_th, float alpha) {
    TORCH_CHECK(grad_y.dtype() == at::kHalf || grad_y.dtype() == at::kFloat);
    TORCH_INTERNAL_ASSERT(grad_y.is_contiguous());
    TORCH_INTERNAL_ASSERT(grad_y.device().type() == at::DeviceType::CUDA);

    TORCH_CHECK(m.dtype() == at::kHalf || m.dtype() == at::kFloat);
    TORCH_INTERNAL_ASSERT(m.is_contiguous());
    TORCH_INTERNAL_ASSERT(m.device().type() == at::DeviceType::CUDA);

    TORCH_CHECK(y.dtype() == at::kHalf || y.dtype() == at::kFloat);
    TORCH_INTERNAL_ASSERT(y.is_contiguous());
    TORCH_INTERNAL_ASSERT(y.device().type() == at::DeviceType::CUDA);

    at::Tensor grad_x = torch::zeros(grad_y.sizes(), grad_y.options());

    int numel = grad_y.numel();
    int B = grad_y.size(0);
    int T = grad_y.size(1);
    int N = numel / (B * T);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_y.scalar_type(), "if_btn_backward_kernel", ([&] {
        if_btn_backward_kernel<scalar_t><<<numel / 1024 + 1, 1024>>>(
                numel, T, N, grad_y.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), m.data_ptr<scalar_t>(), v_th, alpha, grad_x.data_ptr<scalar_t>()
        );
    }));
    return {grad_x};
}

