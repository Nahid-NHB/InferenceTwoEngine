// include/tinyllm/llama_block.hpp
// -----------------------------------------------------------------------------
// One Llama transformer block. Pre-norm variant (the standard Llama 2/3
// recipe):
//
//   h = x + attention(rmsnorm(x), ...)
//   h = h + mlp(rmsnorm(h), ...)
//
// Both sub-blocks are wrapped with residual connections. We pass
// `start_pos` through to attention for RoPE; the MLP doesn't depend on
// position.
// -----------------------------------------------------------------------------
#pragma once

#include "tinyllm/attention.hpp"
#include "tinyllm/mlp.hpp"
#include "tinyllm/rmsnorm.hpp"
#include "tinyllm/tensor.hpp"

namespace tinyllm {

struct LlamaBlockWeights {
    // Two norms, one per sub-block.
    Tensor attn_norm;   // [hidden]
    Tensor mlp_norm;    // [hidden]
    // Attention (Q/K/V/O projections).
    AttentionWeights attn;
    // MLP (gate/up/down).
    MlpWeights mlp;
};

// Run one block: residual + pre-norm attention, then residual + pre-norm MLP.
Tensor llama_block_forward(const Tensor& x,
                           const LlamaBlockWeights& w,
                           const AttentionConfig& cfg,
                           int64_t start_pos = 0);

}  // namespace tinyllm