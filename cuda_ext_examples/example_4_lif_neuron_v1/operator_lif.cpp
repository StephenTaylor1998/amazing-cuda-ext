//
// Created by Hangchi Shen on 24-6-22.
//
#include <torch/extension.h>

#include <vector>

std::vector<at::Tensor> lif_tbn_forward(const at::Tensor x, float tau, float v_th);

std::vector<at::Tensor> lif_btn_forward(const at::Tensor x, float tau, float v_th);

std::vector<at::Tensor> lif_tbn_backward(const at::Tensor grad_y, const at::Tensor m, const at::Tensor y,
                                         float tau, float v_th, float alpha);

std::vector<at::Tensor> lif_btn_backward(const at::Tensor grad_y, const at::Tensor m, const at::Tensor y,
                                         float tau, float v_th, float alpha);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("lif_tbn_forward", &lif_tbn_forward, "lif forward [T, B, N] (GPU)");
    m.def("lif_btn_forward", &lif_btn_forward, "lif forward [B, T, N] (GPU)");
    m.def("lif_tbn_backward", &lif_tbn_backward, "lif backward [T, B, N] (GPU)");
    m.def("lif_btn_backward", &lif_btn_backward, "lif backward [B, T, N] (GPU)");
}
