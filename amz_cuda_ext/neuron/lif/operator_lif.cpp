#include <torch/extension.h>

#include <vector>

std::vector<at::Tensor> forward_lif(const at::Tensor x, float tau, float v_th, float alpha);

std::vector<at::Tensor> backward_lif(const at::Tensor grad_y, const at::Tensor m, const at::Tensor y,
                                     float tau, float v_th, float alpha);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_lif", &forward_lif, "lif forward (GPU)");
    m.def("backward_lif", &backward_lif, "lif backward (GPU)");
}
