// src/sampler.cpp
#include "tinyllm/sampler.hpp"

#include "tinyllm/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace tinyllm {

namespace {

// View logits as a flat 1-D array of length V. Throws on shape error.
struct LogitsView {
    const float* p;
    int64_t v;
    LogitsView(const Tensor& logits) {
        if (logits.dtype() != DType::Float32) {
            throw std::runtime_error("sampler: logits must be Float32");
        }
        int64_t total = 1;
        for (auto d : logits.shape()) total *= d;
        v = total;
        p = logits.data_float();
    }
};

uint64_t make_seed(std::optional<uint64_t> seed) {
    if (seed.has_value()) return *seed;
    std::random_device rd;
    return (static_cast<uint64_t>(rd()) << 32) ^ static_cast<uint64_t>(rd());
}

// Numerically stable softmax returning F32 vector. For sampling we
// don't need the Tensor type machinery.
std::vector<float> softmax_vec(const float* p, int64_t n, float temperature) {
    if (temperature <= 0.0f) {
        // Same as greedy for the purposes of sampling; the caller should
        // have used sample_greedy. But we still softmax with T=1 so the
        // rest of the pipeline is consistent.
        temperature = 1.0f;
    }
    std::vector<float> out(static_cast<std::size_t>(n));
    float maxv = -std::numeric_limits<float>::infinity();
    for (int64_t i = 0; i < n; ++i) {
        float v = p[i] / temperature;
        if (v > maxv) maxv = v;
    }
    double sum = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        double v = std::exp(static_cast<double>(p[i] / temperature) - maxv);
        out[static_cast<std::size_t>(i)] = static_cast<float>(v);
        sum += v;
    }
    float inv = static_cast<float>(1.0 / sum);
    for (int64_t i = 0; i < n; ++i) {
        out[static_cast<std::size_t>(i)] *= inv;
    }
    return out;
}

}  // namespace

int32_t sample_greedy(const Tensor& logits) {
    LogitsView v(logits);
    if (v.v <= 0) throw std::runtime_error("sample_greedy: empty logits");
    int64_t best = 0;
    float   best_val = v.p[0];
    for (int64_t i = 1; i < v.v; ++i) {
        if (v.p[i] > best_val) {
            best_val = v.p[i];
            best = i;
        }
    }
    return static_cast<int32_t>(best);
}

Tensor filter_logits(const Tensor& logits, const SamplerConfig& cfg) {
    LogitsView v(logits);
    int64_t n = v.v;
    Tensor out({n}, DType::Float32);
    float* op = out.data_float();

    if (cfg.temperature <= 0.0f) {
        // Greedy: argmax gets +inf, everything else -inf.
        int64_t best = 0;
        float best_val = v.p[0];
        for (int64_t i = 1; i < n; ++i) {
            if (v.p[i] > best_val) { best_val = v.p[i]; best = i; }
        }
        for (int64_t i = 0; i < n; ++i) {
            op[i] = (i == best) ? std::numeric_limits<float>::infinity()
                                : -std::numeric_limits<float>::infinity();
        }
        return out;
    }

    // Start from a copy.
    for (int64_t i = 0; i < n; ++i) op[i] = v.p[i];

    // Top-k: zero out everything below the k-th largest.
    if (cfg.top_k > 0 && cfg.top_k < n) {
        // Partial-sort: copy to a buffer, find the k-th largest by
        // nth_element, then mask.
        std::vector<float> tmp(op, op + n);
        std::nth_element(tmp.begin(), tmp.begin() + (n - cfg.top_k), tmp.end());
        float threshold = tmp[static_cast<std::size_t>(n - cfg.top_k)];
        // Keep values > threshold, OR == threshold up to k total (rare
        // ties). For simplicity just keep > threshold; top_k isn't
        // meant to be exact under ties.
        for (int64_t i = 0; i < n; ++i) {
            if (op[i] < threshold) {
                op[i] = -std::numeric_limits<float>::infinity();
            }
        }
    }

    // Top-p: zero out everything outside the smallest set whose softmax
    // mass >= p. We compute softmax once, then sort, then mask.
    if (cfg.top_p < 1.0f) {
        auto probs = softmax_vec(op, n, /*temperature=*/1.0f);
        // Sort indices by probability descending.
        std::vector<int64_t> idx(static_cast<std::size_t>(n));
        for (int64_t i = 0; i < n; ++i) idx[static_cast<std::size_t>(i)] = i;
        std::sort(idx.begin(), idx.end(),
                  [&probs](int64_t a, int64_t b) {
                      return probs[static_cast<std::size_t>(a)] >
                             probs[static_cast<std::size_t>(b)];
                  });
        double cum = 0.0;
        int64_t keep = n;  // by default, keep all (when total prob < p)
        for (int64_t k = 0; k < n; ++k) {
            cum += probs[static_cast<std::size_t>(idx[static_cast<std::size_t>(k)])];
            if (cum >= cfg.top_p) { keep = k + 1; break; }
        }
        std::vector<bool> keep_mask(static_cast<std::size_t>(n), false);
        for (int64_t k = 0; k < keep; ++k) {
            keep_mask[static_cast<std::size_t>(idx[static_cast<std::size_t>(k)])] = true;
        }
        for (int64_t i = 0; i < n; ++i) {
            if (!keep_mask[static_cast<std::size_t>(i)]) {
                op[i] = -std::numeric_limits<float>::infinity();
            }
        }
    }
    return out;
}

int32_t sample(const Tensor& logits, const SamplerConfig& cfg,
               std::optional<uint64_t> seed) {
    LogitsView v(logits);

    if (cfg.temperature <= 0.0f || (cfg.top_k == 1 && cfg.top_p >= 1.0f)) {
        return sample_greedy(logits);
    }

    // Greedy via top_k=1 is the same as sample_greedy. But if temperature
    // is non-positive we still go greedy.

    // Apply filtering then softmax + sample. The temperature scales the
    // logits before softmax; filter_logits leaves the values themselves
    // alone (it only masks), so we apply T here.
    Tensor filtered = filter_logits(logits, cfg);
    auto probs = softmax_vec(filtered.data_float(), v.v, cfg.temperature);

    // Inverse-CDF sampling.
    uint64_t s = make_seed(seed);
    std::mt19937_64 rng(s);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    float u = uni(rng);

    double cum = 0.0;
    for (int64_t i = 0; i < v.v; ++i) {
        cum += probs[static_cast<std::size_t>(i)];
        if (u <= cum) return static_cast<int32_t>(i);
    }
    // Numerical tail: pick the last non-zero-prob token.
    for (int64_t i = v.v - 1; i >= 0; --i) {
        if (probs[static_cast<std::size_t>(i)] > 0.0f) {
            return static_cast<int32_t>(i);
        }
    }
    return static_cast<int32_t>(v.v - 1);
}

std::vector<int32_t> generate(LlamaModel& model,
                              const std::vector<int32_t>& prompt,
                              int max_new_tokens,
                              const SamplerConfig& cfg,
                              std::optional<int32_t> eos,
                              std::optional<uint64_t> seed) {
    if (max_new_tokens <= 0) return {};
    if (prompt.empty()) {
        throw std::runtime_error("generate: empty prompt");
    }

    // Prefill the prompt.
    Tensor prompt_t({static_cast<int64_t>(prompt.size())}, DType::Int32);
    for (std::size_t i = 0; i < prompt.size(); ++i) {
        prompt_t.at_flat_int(static_cast<int64_t>(i)) = prompt[i];
    }
    Tensor logits = model.forward_cached(prompt_t, /*start_pos=*/0);

    std::vector<int32_t> generated;
    generated.reserve(static_cast<std::size_t>(max_new_tokens));

    int64_t next_pos = static_cast<int64_t>(prompt.size());
    int64_t vocab = logits.shape()[logits.ndim() - 1];
    // Sample the first token from the last row of prefill logits. Build
    // a fresh [1, vocab] view (as a Tensor copy) so the sample call sees
    // the right row.
    Tensor last_logits({vocab}, DType::Float32);
    std::memcpy(last_logits.data_float(),
                logits.data_float() + (prompt.size() - 1) * vocab,
                static_cast<std::size_t>(vocab) * sizeof(float));
    int32_t next_tok = sample(last_logits, cfg, seed);
    generated.push_back(next_tok);
    if (eos.has_value() && next_tok == *eos) return generated;

    // Decode one token at a time.
    for (int step = 1; step < max_new_tokens; ++step) {
        Tensor one({1}, DType::Int32);
        one.at_flat_int(0) = next_tok;
        Tensor step_logits = model.forward_cached(one, next_pos);
        next_tok = sample(step_logits, cfg, seed);
        generated.push_back(next_tok);
        next_pos += 1;
        if (eos.has_value() && next_tok == *eos) break;
    }
    return generated;
}

}  // namespace tinyllm