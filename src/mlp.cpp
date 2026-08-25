// src/mlp.cpp
#include "tinyllm/mlp.hpp"

#include "tinyllm/matmul.hpp"
#include "tinyllm/tensor.hpp"

#include <cmath>
#include <stdexcept>

namespace tinyllm {

Tensor mlp_forward(const Tensor& x, const MlpWeights& w) {
    if (x.dtype() != DType::Float32) {
        throw std::runtime_error("mlp: x must be Float32");
    }
    if (x.ndim() != 2) {
        throw std::runtime_error("mlp: x must be [seq, hidden]");
    }
    int64_t hidden = x.shape()[1];
    if (w.W_gate.shape()[0] != hidden || w.W_up.shape()[0] != hidden) {
        throw std::runtime_error("mlp: W_gate / W_up must have hidden rows");
    }
    int64_t intermediate = w.W_gate.shape()[1];
    if (w.W_up.shape()[1] != intermediate) {
        throw std::runtime_error("mlp: W_gate and W_up must have same intermediate");
    }
    if (w.W_down.shape()[0] != intermediate || w.W_down.shape()[1] != hidden) {
        throw std::runtime_error("mlp: W_down shape mismatch");
    }

    Tensor gate = ops::matmul(x, w.W_gate);
    Tensor up   = ops::matmul(x, w.W_up);

    // silu(gate) * up
    Tensor silu_gate(gate.shape(), DType::Float32);
    float* sgp = silu_gate.data_float();
    const float* gp = gate.data_float();
    const float* up_p = up.data_float();
    for (int64_t i = 0; i < gate.numel(); ++i) {
        float g = gp[i];
        float silu = g / (1.0f + std::exp(-g));
        sgp[i] = silu * up_p[i];
    }

    return ops::matmul(silu_gate, w.W_down);
}

}  // namespace tinyllm