// include/tinyllm/rope.hpp
// -----------------------------------------------------------------------------
// Rotary Position Embedding (RoPE), Llama flavour (half-rotation).
//
// Given an input tensor x of shape [..., head_dim] where head_dim is even,
// RoPE rotates adjacent pairs of channels using per-position frequencies.
// For head_dim = d and the i-th channel pair (i, i + d/2), the angle is
//
//     theta(p, i) = p * 10000^(-2i / d)
//
// and the rotation applied to (x_i, x_{i+d/2}) is the 2x2 matrix
//
//     [ cos -sin ]
//     [ sin  cos ].
//
// We precompute (cos[p,i], sin[p,i]) tables of shape [seq_len, head_dim/2]
// once per call and apply them to every head in-place.
//
// Two tensor layouts matter for the per-position lookup. We expect either
//   shape = [seq, n_heads, head_dim]   (multi-head self-attention)
// or
//   shape = [seq, n_kv_heads, head_dim] (GQA K/V)
// but in practice we only need to know `seq` (the first dim). The caller
// passes the slice position range so we can precompute just the relevant
// rows.
//
// This is in-place: the input tensor is mutated. Returns void.
//
// Phase 9 may SIMD-optimize the inner pair loop. For now scalar.
// -----------------------------------------------------------------------------
#pragma once

#include "tinyllm/tensor.hpp"

#include <cstdint>

namespace tinyllm {

// Apply RoPE in-place to `x` (shape [seq, ..., head_dim], head_dim even).
// `start_pos` is the position of the first row of `x` (so row p of x uses
// angle for position `start_pos + p`). `theta_base` is the base of the
// geometric schedule (Llama uses 10000).
void rope_inplace(Tensor& x, int64_t start_pos, float theta_base = 10000.0f);

// Precompute the [seq_len, head_dim/2] cos and sin tables.
// Useful when you want to share them across Q and K.
struct RopeTables {
    Tensor cos;  // shape [seq_len, head_dim/2]
    Tensor sin;  // shape [seq_len, head_dim/2]
};
RopeTables precompute_rope_tables(int64_t seq_len, int64_t head_dim,
                                   int64_t start_pos = 0,
                                   float theta_base = 10000.0f);

// Apply RoPE in-place using precomputed tables.
void rope_inplace_with_tables(Tensor& x, const RopeTables& tables);

}  // namespace tinyllm
