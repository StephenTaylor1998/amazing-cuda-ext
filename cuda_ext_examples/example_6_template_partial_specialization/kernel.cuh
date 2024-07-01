#include <type_traits>
#include <torch/extension.h>


struct Function1{};
struct Function2{};

template<typename scalar_t>
__device__ __forceinline__ void add(scalar_t &x) {
    x += 1;
}

template<typename scalar_t>
__device__ __forceinline__ void sub(scalar_t &x) {
    x -= 1;
}

template<typename scalar_t, typename F>
__global__ void kernel(scalar_t *x, int numel) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    if constexpr (std::is_same_v<F, Function1>) add<scalar_t>(x[idx]);
    if constexpr (std::is_same_v<F, Function2>) sub<scalar_t>(x[idx]);
}

template<typename F>
std::vector<torch::Tensor> run_kernel(const torch::Tensor &x) {
    TORCH_CHECK(x.dtype() == at::kHalf || x.dtype() == at::kFloat);
    TORCH_INTERNAL_ASSERT(x.is_contiguous());
    TORCH_INTERNAL_ASSERT(x.device().type() == at::DeviceType::CUDA);

    int numel = x.numel();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "kernel", ([&] {
        kernel<scalar_t, F><<<numel / 256 + 1, 256>>>(x.data_ptr<scalar_t>(), numel);
    }));

    return {x};
}
