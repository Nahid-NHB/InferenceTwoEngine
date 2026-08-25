// tests/test_kv_cache.cpp
// -----------------------------------------------------------------------------
// Phase 6 tests for the KV cache.
//
// Strategy: build a tiny deterministic model, then verify that
//   1. Prefill of N tokens with the cached path matches the non-cached
//      forward on the same N tokens (logits identical).
//   2. Decode (one token at a time, extending the cache) reproduces the
//      non-cached forward's logits at each new position.
//   3. Cache length tracks the number of positions appended.
// -----------------------------------------------------------------------------
#include "test_helpers.hpp"
#include "tinyllm/attention.hpp"
#include "tinyllm/kv_cache.hpp"
#include "tinyllm/llama_block.hpp"
#include "tinyllm/model.hpp"
#include "tinyllm/tensor.hpp"

#include <cmath>
#include <cstring>
#include <vector>

using namespace tinyllm;

namespace {

// Deterministic PRNG so test weights are reproducible.
struct Lcg {
    uint64_t s;
    explicit Lcg(uint64_t seed) : s(seed) {}
    float next() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return (static_cast<float>((s >> 11) & ((1ULL << 21) - 1)) /
                static_cast<float>(1ULL << 21)) * 2.0f - 1.0f;
    }
};

LlamaModelWeights make_tiny_weights(Lcg& rng, const LlamaConfig& cfg) {
    LlamaModelWeights w;
    w.W_embed = Tensor({cfg.vocab_size, cfg.hidden}, DType::Float32);
    for (int64_t i = 0; i < w.W_embed.numel(); ++i) w.W_embed.at_flat(i) = rng.next() * 0.1f;

    w.W_output = Tensor({cfg.hidden, cfg.vocab_size}, DType::Float32);
    for (int64_t i = 0; i < w.W_output.numel(); ++i) w.W_output.at_flat(i) = rng.next() * 0.1f;

    w.final_norm = Tensor({cfg.hidden}, DType::Float32);
    for (int64_t i = 0; i < cfg.hidden; ++i) w.final_norm.at_flat(i) = 1.0f;

    int64_t H   = cfg.hidden;
    int64_t I   = cfg.intermediate;
    int64_t Hd  = cfg.head_dim;
    int64_t KH  = cfg.n_kv_heads;

    w.blocks.resize(static_cast<std::size_t>(cfg.n_layers));
    for (int64_t i = 0; i < cfg.n_layers; ++i) {
        auto& bw = w.blocks[static_cast<std::size_t>(i)].weights;
        bw.attn_norm = Tensor({H}, DType::Float32);
        for (int64_t j = 0; j < H; ++j) bw.attn_norm.at_flat(j) = 1.0f;
        bw.mlp_norm = Tensor({H}, DType::Float32);
        for (int64_t j = 0; j < H; ++j) bw.mlp_norm.at_flat(j) = 1.0f;

        bw.attn.Wq = Tensor({H, H}, DType::Float32);
        bw.attn.Wk = Tensor({H, KH * Hd}, DType::Float32);
        bw.attn.Wv = Tensor({H, KH * Hd}, DType::Float32);
        bw.attn.Wo = Tensor({H, H}, DType::Float32);
        for (int64_t j = 0; j < bw.attn.Wq.numel(); ++j) bw.attn.Wq.at_flat(j) = rng.next() * 0.1f;
        for (int64_t j = 0; j < bw.attn.Wk.numel(); ++j) bw.attn.Wk.at_flat(j) = rng.next() * 0.1f;
        for (int64_t j = 0; j < bw.attn.Wv.numel(); ++j) bw.attn.Wv.at_flat(j) = rng.next() * 0.1f;
        for (int64_t j = 0; j < bw.attn.Wo.numel(); ++j) bw.attn.Wo.at_flat(j) = rng.next() * 0.1f;

        bw.mlp.W_gate = Tensor({H, I}, DType::Float32);
        bw.mlp.W_up   = Tensor({H, I}, DType::Float32);
        bw.mlp.W_down = Tensor({I, H}, DType::Float32);
        for (int64_t j = 0; j < bw.mlp.W_gate.numel(); ++j) bw.mlp.W_gate.at_flat(j) = rng.next() * 0.1f;
        for (int64_t j = 0; j < bw.mlp.W_up.numel();   ++j) bw.mlp.W_up.at_flat(j)   = rng.next() * 0.1f;
        for (int64_t j = 0; j < bw.mlp.W_down.numel(); ++j) bw.mlp.W_down.at_flat(j) = rng.next() * 0.1f;
    }
    return w;
}

}  // namespace

TEST_CASE(kv_cache_construct_and_append) {
    KvCache c(4, 2, 3);
    REQUIRE(c.size() == 0);
    REQUIRE(c.capacity() == 4);
    REQUIRE(c.K.shape()[0] == 4);
    REQUIRE(c.K.shape()[1] == 2);
    REQUIRE(c.K.shape()[2] == 3);

    Tensor K_in({2, 2, 3}, DType::Float32);
    Tensor V_in({2, 2, 3}, DType::Float32);
    for (int64_t i = 0; i < K_in.numel(); ++i) {
        K_in.at_flat(i) = static_cast<float>(i + 1);
        V_in.at_flat(i) = -static_cast<float>(i + 1);
    }
    c.append(K_in, V_in);
    REQUIRE(c.size() == 2);
    REQUIRE(c.K.at_flat(0) == 1.0f);
    REQUIRE(c.K.at_flat(1) == 2.0f);
    REQUIRE(c.V.at_flat(0) == -1.0f);
    // Spot-check the second appended row's last element.
    REQUIRE(c.K.at_flat(11) == 12.0f);

    c.clear();
    REQUIRE(c.size() == 0);

    c.append(K_in, V_in);
    REQUIRE(c.size() == 2);
}

TEST_CASE(kv_cache_capacity_overflow_throws) {
    KvCache c(2, 1, 1);
    Tensor K_in({3, 1, 1}, DType::Float32);
    Tensor V_in({3, 1, 1}, DType::Float32);
    REQUIRE_THROWS(c.append(K_in, V_in));
}

TEST_CASE(kv_cache_shape_mismatch_throws) {
    KvCache c(4, 2, 3);
    Tensor K_in({2, 3, 3}, DType::Float32);
    Tensor V_in({2, 2, 3}, DType::Float32);
    REQUIRE_THROWS(c.append(K_in, V_in));
}

TEST_CASE(cached_prefill_matches_uncached) {
    // Tiny deterministic model.
    LlamaConfig cfg;
    cfg.vocab_size   = 8;
    cfg.hidden       = 4;
    cfg.intermediate = 8;
    cfg.n_heads      = 2;
    cfg.n_kv_heads   = 1;
    cfg.head_dim     = 2;
    cfg.n_layers     = 2;
    cfg.max_seq_len  = 16;

    Lcg rng(42);
    LlamaModelWeights w = make_tiny_weights(rng, cfg);

    LlamaModel m_uncached = make_model(w, cfg);
    LlamaModel m_cached   = make_model(w, cfg);

    Tensor tokens({5}, DType::Int32);
    int32_t ids[5] = {1, 3, 0, 5, 2};
    for (int64_t i = 0; i < 5; ++i) tokens.at_flat_int(i) = ids[i];

    Tensor logits_uncached = m_uncached.forward(tokens);
    Tensor logits_cached   = m_cached.forward_cached(tokens);

    REQUIRE(logits_uncached.shape()[0] == 5);
    REQUIRE(logits_uncached.shape()[1] == cfg.vocab_size);
    REQUIRE(logits_cached.shape()[0] == 5);
    REQUIRE(logits_cached.shape()[1] == cfg.vocab_size);
    for (int64_t i = 0; i < logits_uncached.numel(); ++i) {
        REQUIRE_NEAR(logits_cached.at_flat(i), logits_uncached.at_flat(i), 1e-5f);
    }
    // After prefill the cache should hold all 5 positions.
    REQUIRE(m_cached.caches[0].size() == 5);
}

TEST_CASE(cached_decode_matches_uncached) {
    // The harder test: prefill once, then decode one token at a time.
    // At each decode step, the cached logits for the new token must
    // match what a non-cached forward over [0..t] would have produced
    // at position t.
    LlamaConfig cfg;
    cfg.vocab_size   = 8;
    cfg.hidden       = 4;
    cfg.intermediate = 8;
    cfg.n_heads      = 2;
    cfg.n_kv_heads   = 1;
    cfg.head_dim     = 2;
    cfg.n_layers     = 2;
    cfg.max_seq_len  = 16;

    Lcg rng(99);
    LlamaModelWeights w = make_tiny_weights(rng, cfg);

    LlamaModel m_uncached = make_model(w, cfg);
    LlamaModel m_cached   = make_model(w, cfg);

    int32_t ids[6] = {1, 3, 0, 5, 2, 4};
    Tensor prefill({3}, DType::Int32);
    for (int64_t i = 0; i < 3; ++i) prefill.at_flat_int(i) = ids[i];

    // Prefill both models.
    Tensor _ = m_uncached.forward(prefill);
    Tensor prefill_logits = m_cached.forward_cached(prefill);
    (void)_;

    // The cached prefill should already match the uncached prefill.
    for (int64_t i = 0; i < prefill.numel() * cfg.vocab_size; ++i) {
        REQUIRE_NEAR(prefill_logits.at_flat(i),
                     m_uncached.forward(prefill).at_flat(i), 1e-5f);
    }

    // Now decode tokens 3, 4, 5 one at a time.
    for (int64_t step = 3; step < 6; ++step) {
        Tensor one({1}, DType::Int32);
        one.at_flat_int(0) = ids[step];
        Tensor cached_logits = m_cached.forward_cached(one, /*start_pos=*/step);

        // Reference: run uncached on [0..step+1].
        Tensor all({step + 1}, DType::Int32);
        for (int64_t i = 0; i <= step; ++i) all.at_flat_int(i) = ids[i];
        Tensor ref_logits = m_uncached.forward(all);
        // The last row is what the cached decode produced.
        for (int64_t j = 0; j < cfg.vocab_size; ++j) {
            REQUIRE_NEAR(cached_logits.at_flat(j),
                         ref_logits.at_flat(step * cfg.vocab_size + j),
                         1e-4f);
        }
    }

    // Cache should hold 6 positions.
    REQUIRE(m_cached.caches[0].size() == 6);
}

TEST_CASE(cached_reset_clears_cache) {
    LlamaConfig cfg;
    cfg.vocab_size = 4; cfg.hidden = 4; cfg.intermediate = 4;
    cfg.n_heads = 1; cfg.n_kv_heads = 1; cfg.head_dim = 4; cfg.n_layers = 1;
    cfg.max_seq_len = 8;
    Lcg rng(7);
    LlamaModelWeights w = make_tiny_weights(rng, cfg);
    LlamaModel model = make_model(w, cfg);

    Tensor tok({2}, DType::Int32);
    tok.at_flat_int(0) = 0; tok.at_flat_int(1) = 1;
    model.forward_cached(tok);
    REQUIRE(model.caches[0].size() == 2);

    model.reset_caches();
    REQUIRE(model.caches[0].size() == 0);

    // Should be able to prefill again from position 0.
    Tensor logits = model.forward_cached(tok, /*start_pos=*/0);
    REQUIRE(logits.shape()[0] == 2);
    REQUIRE(model.caches[0].size() == 2);
}

TEST_CASE(cached_capacity_exceeded_throws) {
    LlamaConfig cfg;
    cfg.vocab_size = 4; cfg.hidden = 4; cfg.intermediate = 4;
    cfg.n_heads = 1; cfg.n_kv_heads = 1; cfg.head_dim = 4; cfg.n_layers = 1;
    cfg.max_seq_len = 3;
    Lcg rng(11);
    LlamaModelWeights w = make_tiny_weights(rng, cfg);
    LlamaModel model = make_model(w, cfg);

    Tensor tok({4}, DType::Int32);
    for (int64_t i = 0; i < 4; ++i) tok.at_flat_int(i) = i;
    REQUIRE_THROWS(model.forward_cached(tok));
}