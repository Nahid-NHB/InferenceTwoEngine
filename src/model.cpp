// src/model.cpp
#include "tinyllm/model.hpp"

#include "tinyllm/matmul.hpp"
#include "tinyllm/rmsnorm.hpp"
#include "tinyllm/tensor.hpp"

#include <cstring>
#include <stdexcept>

namespace tinyllm {

Tensor llama_forward(const LlamaModelWeights& w,
                     const Tensor& tokens,
                     const LlamaConfig& cfg,
                     int64_t start_pos) {
    if (tokens.dtype() != DType::Int32) {
        throw std::runtime_error("llama_forward: tokens must be Int32");
    }
    if (tokens.ndim() != 1) {
        throw std::runtime_error("llama_forward: tokens must be 1-D");
    }
    int64_t seq = tokens.shape()[0];
    if (seq <= 0) {
        throw std::runtime_error("llama_forward: empty token sequence");
    }
    if (w.W_embed.dtype() != DType::Float32) {
        throw std::runtime_error("llama_forward: W_embed must be Float32");
    }
    if (w.W_embed.shape()[1] != cfg.hidden) {
        throw std::runtime_error("llama_forward: W_embed hidden mismatch");
    }
    if (w.W_embed.shape()[0] != cfg.vocab_size) {
        throw std::runtime_error("llama_forward: W_embed vocab mismatch");
    }
    if (static_cast<int64_t>(w.blocks.size()) != cfg.n_layers) {
        throw std::runtime_error("llama_forward: wrong number of blocks");
    }
    if (w.W_output.shape()[0] != cfg.hidden ||
        w.W_output.shape()[1] != cfg.vocab_size) {
        throw std::runtime_error("llama_forward: W_output shape mismatch");
    }
    if (w.final_norm.shape()[0] != cfg.hidden) {
        throw std::runtime_error("llama_forward: final_norm shape mismatch");
    }
    AttentionConfig attn_cfg;
    attn_cfg.hidden     = cfg.hidden;
    attn_cfg.n_heads    = cfg.n_heads;
    attn_cfg.n_kv_heads = cfg.n_kv_heads;
    attn_cfg.head_dim   = cfg.head_dim;
    attn_cfg.theta_base = cfg.theta_base;

    // 1) Embedding lookup.
    Tensor h({seq, cfg.hidden}, DType::Float32);
    const int32_t* tp = tokens.data_int();
    const float*   ep = w.W_embed.data_float();
    for (int64_t s = 0; s < seq; ++s) {
        int32_t tok = tp[s];
        if (tok < 0 || tok >= cfg.vocab_size) {
            throw std::runtime_error("llama_forward: token id out of range");
        }
        std::memcpy(h.data_float() + s * cfg.hidden,
                    ep + static_cast<int64_t>(tok) * cfg.hidden,
                    static_cast<std::size_t>(cfg.hidden) * sizeof(float));
    }

    // 2) Stack of blocks.
    for (int64_t layer = 0; layer < cfg.n_layers; ++layer) {
        h = llama_block_forward(h, w.blocks[layer].weights, attn_cfg, start_pos);
    }

    // 3) Final norm.
    h = rmsnorm(h, w.final_norm, cfg.rms_norm_eps);

    // 4) Unembedding.
    return ops::matmul(h, w.W_output);
}

}  // namespace tinyllm