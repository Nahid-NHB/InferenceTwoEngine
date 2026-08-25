// include/tinyllm/sampler.hpp
// -----------------------------------------------------------------------------
// Token sampling.
//
// Given the [seq, vocab] logits output of llama_forward, pick a single
// next token. Four strategies:
//   - greedy:                 argmax(logits)
//   - temperature:            sample from softmax(logits / T)
//   - top-k:                  sample from softmax of (logits with only
//                             the top-k entries kept; rest = -inf)
//   - top-p (nucleus):        sample from softmax of (logits with only
//                             the smallest set of tokens whose
//                             cumulative softmax prob >= p kept)
//
// Strategies compose: temperature sets the sharpness, then top-k and
// top-p filter, then we sample.
//
// Phase 7 keeps the sampler CPU-only and uses std::mt19937 for the
// PRNG. Phase 9 may move it onto SIMD, but for one token per step the
// bottleneck is elsewhere.
// -----------------------------------------------------------------------------
#pragma once

#include "tinyllm/model.hpp"
#include "tinyllm/tensor.hpp"

#include <cstdint>
#include <optional>
#include <vector>

namespace tinyllm {

struct SamplerConfig {
    float temperature = 1.0f;   // <=0 → greedy
    int   top_k       = 0;      // <=0 → disabled
    float top_p       = 1.0f;   // >=1 → disabled
};

// Greedy: returns argmax of the given logits. Ignores temperature and
// filtering. `logits` must be 1-D or [1, vocab]; we flatten.
int32_t sample_greedy(const Tensor& logits);

// Stochastic. If `seed` is nullopt, uses a deterministic-but-unique
// seed derived from std::random_device.
int32_t sample(const Tensor& logits,
               const SamplerConfig& cfg,
               std::optional<uint64_t> seed = std::nullopt);

// Apply temperature + top-k + top-p filtering to logits in-place (well,
// producing a filtered copy as a new tensor). Useful for inspecting
// the post-filter probabilities.
Tensor filter_logits(const Tensor& logits, const SamplerConfig& cfg);

// Generation loop. `prompt` is the initial token sequence (already
// tokenized). The model is updated in-place: its KV caches grow as we
// decode. Stops when max_new_tokens tokens have been generated or an
// EOS token is produced (if `eos` is set). Returns the generated
// sequence (excluding the prompt). Uses `cfg` for sampling.
std::vector<int32_t> generate(LlamaModel& model,
                              const std::vector<int32_t>& prompt,
                              int max_new_tokens,
                              const SamplerConfig& cfg,
                              std::optional<int32_t> eos = std::nullopt,
                              std::optional<uint64_t> seed = std::nullopt);

}  // namespace tinyllm