// include/tinyllm/attention.hpp
// -----------------------------------------------------------------------------
// Grouped-Query Attention (GQA), Llama flavour.
//
// Forward pass:
//   Q = x @ Wq        shape [seq, n_heads   * head_dim]
//   K = x @ Wk        shape [seq, n_kv_heads * head_dim]
//   V = x @ Wv        shape [seq, n_kv_heads * head_dim]
//   reshape Q, K, V to [seq, *, head_dim]
//   rope_inplace(Q, start_pos), rope_inplace(K, start_pos)
//   broadcast K, V from n_kv_heads to n_heads (each kv head serves
//       n_heads / n_kv_heads q heads)
//   scores = Q @ Kᵀ * (1 / sqrt(head_dim))   shape [seq_q, seq_k]
//   mask: causal — set entries where k_pos > q_pos to -inf
//   weights = softmax(scores, axis=-1)
//   ctx = weights @ V                shape [seq_q, n_heads * head_dim]
//   out = ctx @ Wo                   shape [seq_q, hidden]
//
// Phase 5 does the simple version: no KV cache, full causal mask, no
// batched heads (one matmul per head). Phase 6 will add the cache.
// Phase 9 will fold Q @ Kᵀ across heads into one big matmul.
// -----------------------------------------------------------------------------
#pragma once

#include "tinyllm/tensor.hpp"

#include <cstdint>

namespace tinyllm {

struct AttentionWeights {
    Tensor Wq;   // [hidden, n_heads   * head_dim]
    Tensor Wk;   // [hidden, n_kv_heads * head_dim]
    Tensor Wv;   // [hidden, n_kv_heads * head_dim]
    Tensor Wo;   // [n_heads * head_dim, hidden]
};

struct AttentionConfig {
    int64_t hidden;
    int64_t n_heads;
    int64_t n_kv_heads;
    int64_t head_dim;   // = hidden / n_heads (typical)
    float   theta_base = 10000.0f;
};

// Forward pass. `x` is shape [seq, hidden]. `start_pos` is the absolute
// position of the first row of `x` (used for RoPE).
Tensor attention_forward(const Tensor& x,
                         const AttentionWeights& w,
                         const AttentionConfig& cfg,
                         int64_t start_pos = 0);

}  // namespace tinyllm