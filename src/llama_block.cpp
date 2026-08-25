// src/llama_block.cpp
#include "tinyllm/llama_block.hpp"

#include "tinyllm/tensor.hpp"

#include <stdexcept>

namespace tinyllm {

Tensor llama_block_forward(const Tensor& x,
                           const LlamaBlockWeights& w,
                           const AttentionConfig& cfg,
                           int64_t start_pos) {
    if (x.dtype() != DType::Float32) {
        throw std::runtime_error("llama_block: x must be Float32");
    }
    if (x.ndim() != 2) {
        throw std::runtime_error("llama_block: x must be [seq, hidden]");
    }

    // Pre-norm attention with residual.
    Tensor normed = rmsnorm(x, w.attn_norm);
    Tensor attn_out = attention_forward(normed, w.attn, cfg, start_pos);
    Tensor h = x + attn_out;  // ops::operator+

    // Pre-norm MLP with residual.
    Tensor normed2 = rmsnorm(h, w.mlp_norm);
    Tensor mlp_out = mlp_forward(normed2, w.mlp);
    return h + mlp_out;
}

}  // namespace tinyllm