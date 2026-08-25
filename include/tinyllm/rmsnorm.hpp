// include/tinyllm/rmsnorm.hpp
// -----------------------------------------------------------------------------
// RMSNorm: y = (x / RMS(x)) * gamma,   RMS(x) = sqrt(mean(x^2) + eps).
//
// Used as the pre-norm in every Llama block (before attention and before
// the MLP). Cheaper than LayerNorm because there's no mean subtraction
// and no bias — just scale by the per-channel weights `gamma`.
//
// Inputs:
//   x:    [..., d]    (any leading shape; normalization is over the last dim)
//   gamma: [d]         (per-channel scale; shape [d])
//
// Returns: same shape and dtype as `x`. Always F32 for now.
//
// Implementation is a plain loop over the last dim; the inner reduction is
// small (head_dim typically 64–128). We'll come back to SIMD in Phase 9 if
// profiling says it's worth it.
// -----------------------------------------------------------------------------
#pragma once

#include "tinyllm/tensor.hpp"

namespace tinyllm {

Tensor rmsnorm(const Tensor& x, const Tensor& gamma, float eps = 1e-5f);

}  // namespace tinyllm
