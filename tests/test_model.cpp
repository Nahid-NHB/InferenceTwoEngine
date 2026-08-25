// tests/test_model.cpp
#include "test_helpers.hpp"
#include "tinyllm/attention.hpp"
#include "tinyllm/llama_block.hpp"
#include "tinyllm/model.hpp"
#include "tinyllm/tensor.hpp"

#include <cmath>
#include <cstring>

using namespace tinyllm;

namespace {

// Fill a 2D tensor with a small constant.
void fill(Tensor& t, float v) { t.fill(v); }

LlamaBlockWeights make_block_weights(int64_t H, int64_t I, int64_t Hd,
                                     int64_t /*n_heads*/, int64_t n_kv_heads) {
    LlamaBlockWeights bw;
    bw.attn_norm = Tensor({H}, DType::Float32);
    bw.attn_norm.fill(1.0f);
    bw.mlp_norm = Tensor({H}, DType::Float32);
    bw.mlp_norm.fill(1.0f);

    int64_t KVH = n_kv_heads * Hd;
    bw.attn.Wq = Tensor({H, H},         DType::Float32); fill(bw.attn.Wq, 0.0f);
    bw.attn.Wk = Tensor({H, KVH},       DType::Float32); fill(bw.attn.Wk, 0.0f);
    bw.attn.Wv = Tensor({H, KVH},       DType::Float32); fill(bw.attn.Wv, 0.0f);
    bw.attn.Wo = Tensor({H, H},         DType::Float32); fill(bw.attn.Wo, 0.0f);
    // Identity-ish: diagonals = 1.
    for (int64_t i = 0; i < H; ++i) bw.attn.Wq.at_flat(i * H + i) = 1.0f;
    for (int64_t i = 0; i < KVH; ++i) {
        bw.attn.Wk.at_flat(i * KVH + i) = 1.0f;
        bw.attn.Wv.at_flat(i * KVH + i) = 1.0f;
    }
    for (int64_t i = 0; i < H; ++i) bw.attn.Wo.at_flat(i * H + i) = 1.0f;

    bw.mlp.W_gate = Tensor({H, I}, DType::Float32);
    bw.mlp.W_up   = Tensor({H, I}, DType::Float32);
    bw.mlp.W_down = Tensor({I, H}, DType::Float32);
    // All zeros so the MLP contributes nothing — we can test the block
    // math without the MLP polluting the residual.
    fill(bw.mlp.W_gate, 0.0f);
    fill(bw.mlp.W_up,   0.0f);
    fill(bw.mlp.W_down, 0.0f);
    return bw;
}

}  // namespace

TEST_CASE(llama_block_zero_mlp_preserves_residual) {
    // With MLP weights all zero, the block should compute:
    //   h = x + attn(rmsnorm(x))
    // and the MLP path contributes nothing.
    // We then check that the block is finite and shape-preserving.
    AttentionConfig cfg;
    cfg.hidden = 4; cfg.n_heads = 2; cfg.n_kv_heads = 1; cfg.head_dim = 2;
    LlamaBlockWeights bw = make_block_weights(cfg.hidden, /*intermediate=*/8,
                                              cfg.head_dim, cfg.n_heads,
                                              cfg.n_kv_heads);
    Tensor x({2, cfg.hidden}, DType::Float32);
    for (int64_t i = 0; i < x.numel(); ++i) x.at_flat(i) = 0.1f * static_cast<float>(i + 1);

    Tensor out = llama_block_forward(x, bw, cfg, /*start_pos=*/0);
    REQUIRE(out.shape()[0] == 2);
    REQUIRE(out.shape()[1] == cfg.hidden);
    for (int64_t i = 0; i < out.numel(); ++i) {
        float v = out.at_flat(i);
        REQUIRE(!std::isnan(v));
        REQUIRE(!std::isinf(v));
    }
}

TEST_CASE(llama_forward_smoke) {
    // Tiny model: hidden=4, intermediate=8, 2 layers, 2 heads, GQA 2:1.
    LlamaConfig cfg;
    cfg.vocab_size   = 16;
    cfg.hidden       = 4;
    cfg.intermediate = 8;
    cfg.n_heads      = 2;
    cfg.n_kv_heads   = 1;
    cfg.head_dim     = 2;
    cfg.n_layers     = 2;

    LlamaModelWeights w;
    w.W_embed = Tensor({cfg.vocab_size, cfg.hidden}, DType::Float32);
    w.W_embed.fill(0.0f);
    // Set a few embedding rows to small values so they're not all zero.
    for (int64_t v = 0; v < cfg.vocab_size; ++v) {
        w.W_embed.at_flat(v * cfg.hidden + 0) = 0.1f * static_cast<float>(v + 1);
    }
    w.W_output = Tensor({cfg.hidden, cfg.vocab_size}, DType::Float32);
    w.W_output.fill(0.0f);
    w.final_norm = Tensor({cfg.hidden}, DType::Float32);
    w.final_norm.fill(1.0f);

    // Build blocks.
    w.blocks.resize(cfg.n_layers);
    for (int64_t i = 0; i < cfg.n_layers; ++i) {
        w.blocks[i].weights = make_block_weights(cfg.hidden, cfg.intermediate,
                                                 cfg.head_dim, cfg.n_heads,
                                                 cfg.n_kv_heads);
    }

    Tensor tokens({3}, DType::Int32);
    tokens.at_flat_int(0) = 0;
    tokens.at_flat_int(1) = 5;
    tokens.at_flat_int(2) = 15;

    Tensor logits = llama_forward(w, tokens, cfg, /*start_pos=*/0);
    REQUIRE(logits.shape()[0] == 3);
    REQUIRE(logits.shape()[1] == cfg.vocab_size);
    for (int64_t i = 0; i < logits.numel(); ++i) {
        float v = logits.at_flat(i);
        REQUIRE(!std::isnan(v));
        REQUIRE(!std::isinf(v));
    }
}

TEST_CASE(llama_forward_zero_weights_gives_zero_logits) {
    // If every weight is zero, after the final RMSNorm the activations
    // are zero (RMSNorm divides by RMS which is 0 → epsilon-trick gives
    // 0). The matmul against zero W_output gives zero logits.
    LlamaConfig cfg;
    cfg.vocab_size   = 8;
    cfg.hidden       = 4;
    cfg.intermediate = 8;
    cfg.n_heads      = 2;
    cfg.n_kv_heads   = 1;
    cfg.head_dim     = 2;
    cfg.n_layers     = 1;

    LlamaModelWeights w;
    w.W_embed   = Tensor({cfg.vocab_size, cfg.hidden}, DType::Float32);
    w.W_embed.fill(0.0f);
    w.W_output  = Tensor({cfg.hidden, cfg.vocab_size}, DType::Float32);
    w.W_output.fill(0.0f);
    w.final_norm = Tensor({cfg.hidden}, DType::Float32);
    w.final_norm.fill(0.0f);  // gamma=0 → output of RMSNorm is zero
    w.blocks.resize(cfg.n_layers);
    w.blocks[0].weights = make_block_weights(cfg.hidden, cfg.intermediate,
                                             cfg.head_dim, cfg.n_heads,
                                             cfg.n_kv_heads);

    Tensor tokens({2}, DType::Int32);
    tokens.at_flat_int(0) = 1;
    tokens.at_flat_int(1) = 2;
    Tensor logits = llama_forward(w, tokens, cfg, 0);
    for (int64_t i = 0; i < logits.numel(); ++i) {
        REQUIRE_NEAR(logits.at_flat(i), 0.0f, 1e-5f);
    }
}

TEST_CASE(llama_forward_rejects_bad_tokens_dtype) {
    LlamaConfig cfg;
    cfg.vocab_size = 4; cfg.hidden = 4; cfg.intermediate = 4;
    cfg.n_heads = 1; cfg.n_kv_heads = 1; cfg.head_dim = 4; cfg.n_layers = 1;
    LlamaModelWeights w;
    w.W_embed   = Tensor({cfg.vocab_size, cfg.hidden}, DType::Float32);
    w.W_output  = Tensor({cfg.hidden, cfg.vocab_size}, DType::Float32);
    w.final_norm = Tensor({cfg.hidden}, DType::Float32);
    w.blocks.resize(1);
    w.blocks[0].weights = make_block_weights(cfg.hidden, cfg.intermediate,
                                            cfg.head_dim, cfg.n_heads,
                                            cfg.n_kv_heads);
    Tensor tokens({2}, DType::Float32);  // wrong dtype
    REQUIRE_THROWS(llama_forward(w, tokens, cfg, 0));
}

TEST_CASE(llama_forward_rejects_oov_token) {
    LlamaConfig cfg;
    cfg.vocab_size = 4; cfg.hidden = 4; cfg.intermediate = 4;
    cfg.n_heads = 1; cfg.n_kv_heads = 1; cfg.head_dim = 4; cfg.n_layers = 1;
    LlamaModelWeights w;
    w.W_embed   = Tensor({cfg.vocab_size, cfg.hidden}, DType::Float32);
    w.W_output  = Tensor({cfg.hidden, cfg.vocab_size}, DType::Float32);
    w.final_norm = Tensor({cfg.hidden}, DType::Float32);
    w.blocks.resize(1);
    w.blocks[0].weights = make_block_weights(cfg.hidden, cfg.intermediate,
                                            cfg.head_dim, cfg.n_heads,
                                            cfg.n_kv_heads);
    Tensor tokens({2}, DType::Int32);
    tokens.at_flat_int(0) = 0;
    tokens.at_flat_int(1) = 99;  // out of vocab range
    REQUIRE_THROWS(llama_forward(w, tokens, cfg, 0));
}