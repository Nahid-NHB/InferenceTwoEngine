// include/tinyllm/mlp.hpp
// -----------------------------------------------------------------------------
// SwiGLU MLP, Llama flavour.
//
//   gate = x @ W_gate                [seq, intermediate]
//   up   = x @ W_up                  [seq, intermediate]
//   h    = silu(gate) * up           [seq, intermediate]
//   out  = h    @ W_down             [seq, hidden]
//
// silu(x) = x * sigmoid(x)
//
// The intermediate dim is typically (8/3) * hidden rounded to a multiple
// of 256 — but that's a model-config concern, not the MLP's. The MLP just
// uses whatever `intermediate` size the caller passes.
// -----------------------------------------------------------------------------
#pragma once

#include "tinyllm/tensor.hpp"

namespace tinyllm {

struct MlpWeights {
    Tensor W_gate;  // [hidden, intermediate]
    Tensor W_up;    // [hidden, intermediate]
    Tensor W_down;  // [intermediate, hidden]
};

Tensor mlp_forward(const Tensor& x, const MlpWeights& w);

}  // namespace tinyllm