// include/tinyllm/model.hpp
// -----------------------------------------------------------------------------
// Top-level Llama model.
//
//   tokens [seq] -> embedding -> blocks -> final norm -> logits [seq, vocab]
//
// Phase 5 wires the components together. Phase 6 adds:
//   - per-layer KvCache state, owned by LlamaModel
//   - llama_forward_cached(): uses the cache, supports incremental decode
//   - generate_one(): runs one token of prefill/decode given a starting
//     cache position
//
// Embedding: we use a learned [vocab, hidden] matrix W_embed.
// Unembedding (the language-model head): the output projection W_output
// maps hidden -> vocab. Llama 2/3 ties these (W_output == W_embedᵀ) but
// we keep them separate for clarity in Phase 5.
// -----------------------------------------------------------------------------
#pragma once

#include "tinyllm/attention.hpp"
#include "tinyllm/kv_cache.hpp"
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
    int64_t max_seq_len      = 2048;
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

// Holds the weights + per-layer caches. Built from LlamaModelWeights by
// `make_model`, which allocates caches sized for cfg.max_seq_len.
struct LlamaModel {
    LlamaModelWeights weights;
    std::vector<KvCache> caches;       // one per layer
    LlamaConfig cfg;

    // Reset all caches to length 0. Cheap (just sets a counter).
    void reset_caches() noexcept;

    // Pure forward (no cache). Equivalent to running prefill over the
    // entire sequence in one shot.
    Tensor forward(const Tensor& tokens, int64_t start_pos = 0);

    // Cached forward. Appends new K/V rows to each layer's cache and
    // returns logits for the new tokens only.
    Tensor forward_cached(const Tensor& tokens, int64_t start_pos = 0);
};

// Build a LlamaModel from a weights blob and a config. Allocates per-
// layer caches.
LlamaModel make_model(LlamaModelWeights w, const LlamaConfig& cfg);

// One-shot prefill + decode step helper. `tokens` is 1-D Int32. Returns
// the [seq, vocab] logits (Phase 7 will add sampling). Same as
// forward_cached; provided so callers have a clear name.
Tensor llama_forward_cached(const LlamaModel& m,
                            const Tensor& tokens,
                            int64_t start_pos = 0);

// Pure (non-cached) forward. Equivalent to LlamaModel::forward.
// Kept as a free function for backward compatibility with Phase 5 callers.
Tensor llama_forward(const LlamaModelWeights& w,
                     const Tensor& tokens,
                     const LlamaConfig& cfg,
                     int64_t start_pos = 0);

}  // namespace tinyllm