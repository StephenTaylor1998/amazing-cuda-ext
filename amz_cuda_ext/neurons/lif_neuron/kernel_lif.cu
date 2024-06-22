#include <torch/extension.h>
#include <vector>

__global__ void lif_tbn_forward_kernel(int T, int BN, const float *x, const float tau, const float v_th,
                                       const float alpha, float *y, float *m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float v = 0.0f;
    if (idx < BN) {
        for (int t = 0; t < T; t++) {
            m[idx + t * BN] = v * tau + x[idx + t * BN];
            y[idx + t * BN] = (m[idx + t * BN] - v_th) >= 0.0f ? 1.0f : 0.0f;
            v = m[idx + t * BN] * (1 - y[idx + t * BN]);
        }
    }
}

__global__ void lif_btn_forward_kernel(int numel, int T, int N, const float *x, const float tau, const float v_th,
                                       const float alpha, float *y, float *m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    float v = 0.0f;
    if ((idx % (N * T)) < N) {
        for (int t = 0; t < T; t++) {
            m[idx + t * N] = v * tau + x[idx + t * N];
            y[idx + t * N] = (m[idx + t * N] - v_th) >= 0.0f ? 1.0f : 0.0f;
            v = m[idx + t * N] * (1 - y[idx + t * N]);
        }
    }
}

__global__ void lif_tbn_backward_kernel(int T, int BN, const float *grad_y, const float *y, const float *m,
                                        const float tau, const float v_th, const float alpha, float *grad_x) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float grad_v = 0.0f;
    if (idx < BN) {
        for (int t = T - 1; t >= 0; t--) {
            float grad_sg_1 = (alpha - abs(m[idx + t * BN] - v_th));
            float clamp = grad_sg_1 < 0.0f ? 0.0f : grad_sg_1;
            float grad_sg = clamp * (1. / alpha) * (1. / alpha);
            float grad_m = grad_y[idx + t * BN] * grad_sg + grad_v * (1. - m[idx + t * BN] * grad_sg - y[idx + t * BN]);
            grad_x[idx + t * BN] = grad_m;
            grad_v = grad_m * tau;
        }
    }
}

__global__ void lif_btn_backward_kernel(int numel, int T, int N, const float *grad_y, const float *y, const float *m,
                                        const float tau, const float v_th, const float alpha, float *grad_x) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    float grad_v = 0.0f;
    if ((idx % (N * T)) < N) {
        for (int t = T - 1; t >= 0; t--) {
            float grad_sg_1 = (alpha - abs(m[idx + t * N] - v_th));
            float clamp = grad_sg_1 < 0.0f ? 0.0f : grad_sg_1;
            float grad_sg = clamp * (1. / alpha) * (1. / alpha);
            float grad_m = grad_y[idx + t * N] * grad_sg + grad_v * (1. - m[idx + t * N] * grad_sg - y[idx + t * N]);
            grad_x[idx + t * N] = grad_m;
            grad_v = grad_m * tau;
        }
    }
}

std::vector <at::Tensor> lif_tbn_forward(const at::Tensor x, float tau, float v_th, float alpha) {
    TORCH_CHECK(x.dtype() == at::kFloat);
    TORCH_INTERNAL_ASSERT(x.is_contiguous());
    TORCH_INTERNAL_ASSERT(x.device().type() == at::DeviceType::CUDA);

    at::Tensor y = torch::empty(x.sizes(), x.options());
    at::Tensor m = torch::empty(x.sizes(), x.options());

    int numel = x.numel();
    int T = x.size(0);
    int BN = numel / T;

    lif_tbn_forward_kernel<<<numel / 256 + 1, 256>>>(T, BN, x.data_ptr<float>(), tau, v_th, alpha,
            y.data_ptr<float>(), m.data_ptr<float>());
//    lif_btn_forward_kernel<<<numel / 256 + 1, 256>>>(numel, T, BN, x.data_ptr<float>(), tau, v_th, alpha,
//                                                     y.data_ptr<float>(), m.data_ptr<float>());

    return {m, y};
}

std::vector <at::Tensor> lif_btn_forward(const at::Tensor x, float tau, float v_th, float alpha) {
    TORCH_CHECK(x.dtype() == at::kFloat);
    TORCH_INTERNAL_ASSERT(x.is_contiguous());
    TORCH_INTERNAL_ASSERT(x.device().type() == at::DeviceType::CUDA);

    at::Tensor y = torch::empty(x.sizes(), x.options());
    at::Tensor m = torch::empty(x.sizes(), x.options());

    int numel = x.numel();
    int B = x.size(0);
    int T = x.size(1);
    int N = numel / (B * T);

    lif_btn_forward_kernel<<<numel / 256 + 1, 256>>>(numel, T, N, x.data_ptr<float>(), tau, v_th, alpha,
            y.data_ptr<float>(), m.data_ptr<float>());

    return {m, y};
}

std::vector <at::Tensor> lif_tbn_backward(const at::Tensor grad_y, const at::Tensor m, const at::Tensor y,
                                          float tau, float v_th, float alpha) {
    TORCH_CHECK(grad_y.dtype() == at::kFloat);
    TORCH_INTERNAL_ASSERT(grad_y.is_contiguous());
    TORCH_INTERNAL_ASSERT(grad_y.device().type() == at::DeviceType::CUDA);

    TORCH_CHECK(m.dtype() == at::kFloat);
    TORCH_INTERNAL_ASSERT(m.is_contiguous());
    TORCH_INTERNAL_ASSERT(m.device().type() == at::DeviceType::CUDA);

    TORCH_CHECK(y.dtype() == at::kFloat);
    TORCH_INTERNAL_ASSERT(y.is_contiguous());
    TORCH_INTERNAL_ASSERT(y.device().type() == at::DeviceType::CUDA);

    at::Tensor grad_x = torch::empty(grad_y.sizes(), grad_y.options());

    int numel = grad_y.numel();
    int T = grad_y.size(0);
    int BN = numel / T;

    lif_tbn_backward_kernel<<<numel / 256 + 1, 256>>>(T, BN, grad_y.data_ptr<float>(), y.data_ptr<float>(),
            m.data_ptr<float>(), tau, v_th, alpha, grad_x.data_ptr<float>());
//    lif_btn_backward_kernel<<<numel / 256 + 1, 256>>>(numel, T, BN, grad_y.data_ptr<float>(), y.data_ptr<float>(),
//                                                      m.data_ptr<float>(), tau, v_th, alpha, grad_x.data_ptr<float>());

    return {grad_x};
}

std::vector <at::Tensor> lif_btn_backward(const at::Tensor grad_y, const at::Tensor m, const at::Tensor y,
                                          float tau, float v_th, float alpha) {
    TORCH_CHECK(grad_y.dtype() == at::kFloat);
    TORCH_INTERNAL_ASSERT(grad_y.is_contiguous());
    TORCH_INTERNAL_ASSERT(grad_y.device().type() == at::DeviceType::CUDA);

    TORCH_CHECK(m.dtype() == at::kFloat);
    TORCH_INTERNAL_ASSERT(m.is_contiguous());
    TORCH_INTERNAL_ASSERT(m.device().type() == at::DeviceType::CUDA);

    TORCH_CHECK(y.dtype() == at::kFloat);
    TORCH_INTERNAL_ASSERT(y.is_contiguous());
    TORCH_INTERNAL_ASSERT(y.device().type() == at::DeviceType::CUDA);

    at::Tensor grad_x = torch::empty(grad_y.sizes(), grad_y.options());

    int numel = grad_y.numel();
    int B = grad_y.size(0);
    int T = grad_y.size(1);
    int N = numel / (B * T);

    lif_btn_backward_kernel<<<numel / 256 + 1, 256>>>(numel, T, N, grad_y.data_ptr<float>(), y.data_ptr<float>(),
            m.data_ptr<float>(), tau, v_th, alpha, grad_x.data_ptr<float>());

    return {grad_x};
}

