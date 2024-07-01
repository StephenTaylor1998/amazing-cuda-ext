//
// Created by Hangchi Shen on 24-6-28.
//
#include <torch/extension.h>
#include <vector>
#include "kernel.cuh"

template<typename scalar_t>
std::vector<at::Tensor> run_kernel(at::Tensor x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &run_kernel<Function1>, "add (GPU)");
    m.def("sub", &run_kernel<Function2>, "sub (GPU)");
}
