// src/model.cpp
#include "tinyllm/model.hpp"

#include "tinyllm/matmul.hpp"
#include "tinyllm/rmsnorm.hpp"
#include "tinyllm/tensor.hpp"

#include <cstring>
#include <stdexcept>

namespace tinyllm {

namespace {

void validate_inputs(const LlamaModelWeights& w, const LlamaConfig& cfg) {
    if (w.W_embed.dtype() != DType::Float32) {
        throw std::runtime_error("llama: W_embed must be Float32");
    }
    if (w.W_embed.shape()[0] != cfg.vocab_size ||
        w.W_embed.shape()[1] != cfg.hidden) {
        throw std::runtime_error("llama: W_embed shape mismatch");
    }
    if (static_cast<int64_t>(w.blocks.size()) != cfg.n_layers) {
        throw std::runtime_error("llama: wrong number of blocks");
    }
    if (w.W_output.shape()[0] != cfg.hidden ||
        w.W_output.shape()[1] != cfg.vocab_size) {
        throw std::runtime_error("llama: W_output shape mismatch");
    }
    if (w.final_norm.shape()[0] != cfg.hidden) {
        throw std::runtime_error("llama: final_norm shape mismatch");
    }
}

AttentionConfig make_attn_cfg(const LlamaConfig& cfg) {
    AttentionConfig a;
    a.hidden     = cfg.hidden;
    a.n_heads    = cfg.n_heads;
    a.n_kv_heads = cfg.n_kv_heads;
    a.head_dim   = cfg.head_dim;
    a.theta_base = cfg.theta_base;
    return a;
}

Tensor embed(const LlamaModelWeights& w, const Tensor& tokens,
             const LlamaConfig& cfg) {
    if (tokens.dtype() != DType::Int32) {
        throw std::runtime_error("llama: tokens must be Int32");
    }
    if (tokens.ndim() != 1) {
        throw std::runtime_error("llama: tokens must be 1-D");
    }
    int64_t seq = tokens.shape()[0];
    if (seq <= 0) {
        throw std::runtime_error("llama: empty token sequence");
    }
    Tensor h({seq, cfg.hidden}, DType::Float32);
    const int32_t* tp = tokens.data_int();
    const float*   ep = w.W_embed.data_float();
    for (int64_t s = 0; s < seq; ++s) {
        int32_t tok = tp[s];
        if (tok < 0 || tok >= cfg.vocab_size) {
            throw std::runtime_error("llama: token id out of range");
        }
        std::memcpy(h.data_float() + s * cfg.hidden,
                    ep + static_cast<int64_t>(tok) * cfg.hidden,
                    static_cast<std::size_t>(cfg.hidden) * sizeof(float));
    }
    return h;
}

Tensor run_blocks(const LlamaModelWeights& w, const LlamaConfig& cfg,
                  Tensor h, int64_t start_pos, std::vector<KvCache>* caches) {
    AttentionConfig attn_cfg = make_attn_cfg(cfg);
    for (int64_t layer = 0; layer < cfg.n_layers; ++layer) {
        if (caches) {
            h = llama_block_forward_cached(h, w.blocks[layer].weights,
                                            attn_cfg,
                                            (*caches)[static_cast<std::size_t>(layer)],
                                            start_pos);
        } else {
            h = llama_block_forward(h, w.blocks[layer].weights, attn_cfg, start_pos);
        }
    }
    return h;
}

}  // namespace

void LlamaModel::reset_caches() noexcept {
    for (auto& c : caches) c.clear();
}

Tensor LlamaModel::forward(const Tensor& tokens, int64_t start_pos) {
    validate_inputs(weights, cfg);
    Tensor h = embed(weights, tokens, cfg);
    h = run_blocks(weights, cfg, std::move(h), start_pos, /*caches=*/nullptr);
    h = rmsnorm(h, weights.final_norm, cfg.rms_norm_eps);
    return ops::matmul(h, weights.W_output);
}

Tensor LlamaModel::forward_cached(const Tensor& tokens, int64_t start_pos) {
    validate_inputs(weights, cfg);
    Tensor h = embed(weights, tokens, cfg);
    h = run_blocks(weights, cfg, std::move(h), start_pos, &caches);
    h = rmsnorm(h, weights.final_norm, cfg.rms_norm_eps);
    return ops::matmul(h, weights.W_output);
}

LlamaModel make_model(LlamaModelWeights w, const LlamaConfig& cfg) {
    validate_inputs(w, cfg);
    LlamaModel m;
    m.weights = std::move(w);
    m.cfg = cfg;
    m.caches.reserve(static_cast<std::size_t>(cfg.n_layers));
    for (int64_t i = 0; i < cfg.n_layers; ++i) {
        m.caches.emplace_back(cfg.max_seq_len, cfg.n_kv_heads, cfg.head_dim);
    }
    return m;
}

Tensor llama_forward_cached(const LlamaModel& m, const Tensor& tokens,
                            int64_t start_pos) {
    // const_cast because the cache must be mutated (LlamaModel caches
    // are part of the model state, which is the user's, but in Phase 6
    // we keep the API non-const-correct for simplicity).
    return const_cast<LlamaModel&>(m).forward_cached(tokens, start_pos);
}

// -----------------------------------------------------------------------------
// Phase 5 entry point — kept as a thin wrapper around LlamaModel::forward
// for backward compatibility with the Phase 5 tests.
// -----------------------------------------------------------------------------
Tensor llama_forward(const LlamaModelWeights& w,
                     const Tensor& tokens,
                     const LlamaConfig& cfg,
                     int64_t start_pos) {
    LlamaModel m = make_model(const_cast<LlamaModelWeights&>(w), cfg);
    return m.forward(tokens, start_pos);
}

}  // namespace tinyllm