// include/tinyllm/model.hpp
// -----------------------------------------------------------------------------
// Top-level Llama model.
//
//   tokens [seq] -> embedding -> blocks -> final norm -> logits [seq, vocab]
//
// Phase 5 wires the components together. Phase 6 will add the KV cache for
// efficient autoregressive decoding.
//
// Embedding: we use a learned [vocab, hidden] matrix W_embed.
// Unembedding (the language-model head): the output projection W_output
// maps hidden -> vocab. Llama 2/3 ties these (W_output == W_embedᵀ) but
// we keep them separate for clarity in Phase 5.
// -----------------------------------------------------------------------------
#pragma once

#include "tinyllm/attention.hpp"
#include "tinyllm/llama_block.hpp"
#include "tinyllm/tensor.hpp"

#include <cstdint>
#include <vector>

namespace tinyllm {

struct LlamaConfig {
    int64_t vocab_size       = 32000;
    int64_t hidden           = 4096;
    int64_t intermediate     = 11008;
    int64_t n_heads          = 32;
    int64_t n_kv_heads       = 32;
    int64_t head_dim         = 128;     // hidden / n_heads
    int64_t n_layers         = 32;
    float   rms_norm_eps     = 1e-5f;
    float   theta_base       = 10000.0f;
};

struct LlamaBlock {
    LlamaBlockWeights weights;
};

struct LlamaModelWeights {
    Tensor W_embed;          // [vocab, hidden]
    std::vector<LlamaBlock> blocks;   // n_layers blocks
    Tensor final_norm;       // [hidden]
    Tensor W_output;         // [hidden, vocab]
};

// One forward pass. `tokens` is a 1-D int32 tensor of token ids.
Tensor llama_forward(const LlamaModelWeights& w,
                     const Tensor& tokens,
                     const LlamaConfig& cfg,
                     int64_t start_pos = 0);

}  // namespace tinyllm