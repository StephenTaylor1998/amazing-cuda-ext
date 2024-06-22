#include <torch/extension.h>

#include <vector>

std::vector<at::Tensor> if_tbn_forward(const at::Tensor x, float v_th, float alpha);

std::vector<at::Tensor> if_btn_forward(const at::Tensor x, float v_th, float alpha);

std::vector<at::Tensor> if_tbn_backward(const at::Tensor grad_y, const at::Tensor m, const at::Tensor y,
                                        float v_th, float alpha);

std::vector<at::Tensor> if_btn_backward(const at::Tensor grad_y, const at::Tensor m, const at::Tensor y,
                                        float v_th, float alpha);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("if_tbn_forward", &if_tbn_forward, "if forward [T, B, N] (GPU)");
    m.def("if_btn_forward", &if_btn_forward, "if forward [B, T, N] (GPU)");
    m.def("if_tbn_backward", &if_tbn_backward, "if backward [T, B, N] (GPU)");
    m.def("if_btn_backward", &if_btn_backward, "if backward [B, T, N] (GPU)");
}
