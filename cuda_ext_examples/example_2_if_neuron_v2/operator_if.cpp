//
// Created by Hangchi Shen on 24-6-22.
//
#include <torch/extension.h>

#include <vector>

std::vector<at::Tensor> if_tbn_forward(at::Tensor x, float v_th);

std::vector<at::Tensor> if_btn_forward(at::Tensor x, float v_th);

std::vector<at::Tensor> if_tbn_backward(at::Tensor grad_y, at::Tensor m, at::Tensor y,
                                        float v_th, float alpha);

std::vector<at::Tensor> if_btn_backward(at::Tensor grad_y, at::Tensor m, at::Tensor y,
                                        float v_th, float alpha);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("if_tbn_forward", &if_tbn_forward, "if forward [T, B, N] (GPU)");
m.def("if_btn_forward", &if_btn_forward, "if forward [B, T, N] (GPU)");
m.def("if_tbn_backward", &if_tbn_backward, "if backward [T, B, N] (GPU)");
m.def("if_btn_backward", &if_btn_backward, "if backward [B, T, N] (GPU)");
}
