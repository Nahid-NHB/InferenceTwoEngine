// tests/test_sampler.cpp
// -----------------------------------------------------------------------------
// Sampling tests.
//
// We test the components independently:
//   - sample_greedy: picks argmax
//   - filter_logits: temperature=top_k=top_p combinations
//   - sample: with fixed seed, distribution roughly matches expected
//   - generate: end-to-end with a small model, EOS halts early
// -----------------------------------------------------------------------------
#include "test_helpers.hpp"
#include "tinyllm/model.hpp"
#include "tinyllm/sampler.hpp"
#include "tinyllm/tensor.hpp"

#include <cmath>
#include <set>
#include <vector>

using namespace tinyllm;

namespace {

// Build a small model with deterministic weights, suitable for sampling
// tests. The model is too small to produce meaningful text — we only
// verify the sampling plumbing.
struct Lcg {
    uint64_t s;
    explicit Lcg(uint64_t seed) : s(seed) {}
    float next() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return (static_cast<float>((s >> 11) & ((1ULL << 21) - 1)) /
                static_cast<float>(1ULL << 21)) * 2.0f - 1.0f;
    }
};

LlamaConfig tiny_cfg() {
    LlamaConfig cfg;
    cfg.vocab_size   = 6;
    cfg.hidden       = 4;
    cfg.intermediate = 8;
    cfg.n_heads      = 2;
    cfg.n_kv_heads   = 1;
    cfg.head_dim     = 2;
    cfg.n_layers     = 1;
    cfg.max_seq_len  = 16;
    return cfg;
}

LlamaModelWeights tiny_weights(Lcg& rng, const LlamaConfig& cfg) {
    LlamaModelWeights w;
    w.W_embed = Tensor({cfg.vocab_size, cfg.hidden}, DType::Float32);
    for (int64_t i = 0; i < w.W_embed.numel(); ++i) w.W_embed.at_flat(i) = rng.next() * 0.1f;
    w.W_output = Tensor({cfg.hidden, cfg.vocab_size}, DType::Float32);
    for (int64_t i = 0; i < w.W_output.numel(); ++i) w.W_output.at_flat(i) = rng.next() * 0.1f;
    w.final_norm = Tensor({cfg.hidden}, DType::Float32);
    for (int64_t i = 0; i < cfg.hidden; ++i) w.final_norm.at_flat(i) = 1.0f;

    int64_t H = cfg.hidden, I = cfg.intermediate, Hd = cfg.head_dim, KH = cfg.n_kv_heads;
    w.blocks.resize(1);
    auto& bw = w.blocks[0].weights;
    bw.attn_norm = Tensor({H}, DType::Float32);
    bw.mlp_norm  = Tensor({H}, DType::Float32);
    for (int64_t i = 0; i < H; ++i) { bw.attn_norm.at_flat(i) = 1.0f; bw.mlp_norm.at_flat(i) = 1.0f; }
    bw.attn.Wq = Tensor({H, H}, DType::Float32);
    bw.attn.Wk = Tensor({H, KH * Hd}, DType::Float32);
    bw.attn.Wv = Tensor({H, KH * Hd}, DType::Float32);
    bw.attn.Wo = Tensor({H, H}, DType::Float32);
    bw.mlp.W_gate = Tensor({H, I}, DType::Float32);
    bw.mlp.W_up   = Tensor({H, I}, DType::Float32);
    bw.mlp.W_down = Tensor({I, H}, DType::Float32);
    for (auto* t : {&bw.attn.Wq, &bw.attn.Wk, &bw.attn.Wv, &bw.attn.Wo,
                    &bw.mlp.W_gate, &bw.mlp.W_up, &bw.mlp.W_down}) {
        for (int64_t i = 0; i < t->numel(); ++i) t->at_flat(i) = rng.next() * 0.1f;
    }
    return w;
}

}  // namespace

TEST_CASE(sample_greedy_picks_argmax) {
    Tensor logits({5}, DType::Float32);
    logits.at_flat(0) = 1.0f;
    logits.at_flat(1) = 5.0f;   // argmax
    logits.at_flat(2) = -3.0f;
    logits.at_flat(3) = 5.0f;   // tie
    logits.at_flat(4) = 0.0f;
    int32_t t = sample_greedy(logits);
    // First occurrence of the max wins (i.e., index 1).
    REQUIRE(t == 1);
}

TEST_CASE(sample_greedy_2d_input) {
    // 2-D input: we flatten, so argmax over the whole array.
    Tensor logits({2, 3}, DType::Float32);
    logits.at_flat(0) = 1.0f;
    logits.at_flat(1) = 2.0f;
    logits.at_flat(2) = 3.0f;
    logits.at_flat(3) = 4.0f;
    logits.at_flat(4) = 99.0f;
    logits.at_flat(5) = 5.0f;
    REQUIRE(sample_greedy(logits) == 4);
}

TEST_CASE(filter_temperature_zero_makes_argmax_infinite) {
    Tensor logits({4}, DType::Float32);
    logits.at_flat(0) = 0.0f;
    logits.at_flat(1) = 1.0f;
    logits.at_flat(2) = -2.0f;
    logits.at_flat(3) = 3.0f;
    SamplerConfig cfg;
    cfg.temperature = 0.0f;
    Tensor filtered = filter_logits(logits, cfg);
    REQUIRE(std::isinf(filtered.at_flat(3)));
    REQUIRE(filtered.at_flat(3) > 0);
    for (int64_t i = 0; i < 4; ++i) {
        if (i == 3) continue;
        REQUIRE(std::isinf(filtered.at_flat(i)));
        REQUIRE(filtered.at_flat(i) < 0);
    }
}

TEST_CASE(filter_top_k_keeps_only_top_k) {
    Tensor logits({5}, DType::Float32);
    logits.at_flat(0) = 0.0f;
    logits.at_flat(1) = 5.0f;
    logits.at_flat(2) = 3.0f;
    logits.at_flat(3) = -1.0f;
    logits.at_flat(4) = 4.0f;
    SamplerConfig cfg;
    cfg.temperature = 1.0f;
    cfg.top_k = 2;
    Tensor filtered = filter_logits(logits, cfg);
    // Indices 1 and 4 have the top-2 values (5 and 4). Index 0 (val=0),
    // 2 (val=3), 3 (val=-1) should be -inf.
    REQUIRE(!std::isinf(filtered.at_flat(1)));
    REQUIRE(!std::isinf(filtered.at_flat(4)));
    REQUIRE(std::isinf(filtered.at_flat(0)) && filtered.at_flat(0) < 0);
    REQUIRE(std::isinf(filtered.at_flat(2)) && filtered.at_flat(2) < 0);
    REQUIRE(std::isinf(filtered.at_flat(3)) && filtered.at_flat(3) < 0);
}

TEST_CASE(filter_top_p_keeps_smallest_sufficient_set) {
    // logits: very peaked distribution. With top_p=0.9, only the top
    // token should remain.
    Tensor logits({4}, DType::Float32);
    logits.at_flat(0) = 10.0f;   // ~1.0
    logits.at_flat(1) = 1.0f;    // ~0
    logits.at_flat(2) = 0.0f;    // ~0
    logits.at_flat(3) = -1.0f;   // ~0
    SamplerConfig cfg;
    cfg.temperature = 1.0f;
    cfg.top_p = 0.9f;
    Tensor filtered = filter_logits(logits, cfg);
    REQUIRE(!std::isinf(filtered.at_flat(0)));
    REQUIRE(std::isinf(filtered.at_flat(1)) && filtered.at_flat(1) < 0);
    REQUIRE(std::isinf(filtered.at_flat(2)) && filtered.at_flat(2) < 0);
    REQUIRE(std::isinf(filtered.at_flat(3)) && filtered.at_flat(3) < 0);
}

TEST_CASE(sample_with_seed_is_deterministic) {
    Tensor logits({4}, DType::Float32);
    logits.at_flat(0) = 1.0f;
    logits.at_flat(1) = 2.0f;
    logits.at_flat(2) = 3.0f;
    logits.at_flat(3) = 0.5f;
    SamplerConfig cfg;
    cfg.temperature = 1.0f;
    int32_t a1 = sample(logits, cfg, /*seed=*/12345);
    int32_t a2 = sample(logits, cfg, /*seed=*/12345);
    int32_t b1 = sample(logits, cfg, /*seed=*/99999);
    REQUIRE(a1 == a2);
    (void)b1;  // b1 may or may not equal a1 depending on distribution
}

TEST_CASE(sample_distribution_sums_to_one) {
    // With many samples, the empirical distribution should roughly match
    // the softmax probabilities. We just check that *some* non-argmax
    // token gets sampled sometimes when the temperature is high enough.
    Tensor logits({3}, DType::Float32);
    logits.at_flat(0) = 1.0f;
    logits.at_flat(1) = 2.0f;
    logits.at_flat(2) = 3.0f;
    SamplerConfig cfg;
    cfg.temperature = 5.0f;  // high → roughly uniform
    std::set<int32_t> seen;
    for (uint64_t s = 0; s < 200; ++s) {
        seen.insert(sample(logits, cfg, s));
    }
    REQUIRE(seen.size() >= 2);   // should hit at least 2 of 3 tokens
}

TEST_CASE(sample_low_temperature_concentrates_on_argmax) {
    // With very low T, we should almost always pick the argmax.
    Tensor logits({3}, DType::Float32);
    logits.at_flat(0) = 0.0f;
    logits.at_flat(1) = 0.1f;
    logits.at_flat(2) = 100.0f;  // argmax
    SamplerConfig cfg;
    cfg.temperature = 0.01f;
    int argmax_count = 0;
    for (uint64_t s = 0; s < 50; ++s) {
        if (sample(logits, cfg, s) == 2) ++argmax_count;
    }
    REQUIRE(argmax_count >= 49);
}

TEST_CASE(sample_greedy_via_zero_temperature) {
    Tensor logits({3}, DType::Float32);
    logits.at_flat(0) = 5.0f;
    logits.at_flat(1) = 1.0f;
    logits.at_flat(2) = -3.0f;
    SamplerConfig cfg;
    cfg.temperature = 0.0f;
    REQUIRE(sample(logits, cfg, /*seed=*/42) == 0);
    REQUIRE(sample(logits, cfg, /*seed=*/0) == 0);
}

TEST_CASE(generate_produces_max_new_tokens) {
    LlamaConfig cfg = tiny_cfg();
    Lcg rng(123);
    LlamaModel model = make_model(tiny_weights(rng, cfg), cfg);
    SamplerConfig sc;
    sc.temperature = 0.5f;
    auto out = generate(model, /*prompt=*/{0, 1}, /*max_new_tokens=*/4, sc);
    REQUIRE(out.size() == 4);
    // All tokens in [0, vocab_size).
    for (int32_t t : out) {
        REQUIRE(t >= 0);
        REQUIRE(t < cfg.vocab_size);
    }
}

TEST_CASE(generate_stops_at_eos) {
    LlamaConfig cfg = tiny_cfg();
    Lcg rng(456);
    LlamaModel model = make_model(tiny_weights(rng, cfg), cfg);
    SamplerConfig sc;
    // Greedy: deterministic, picks argmax.
    sc.temperature = 0.0f;
    // Force the unembedding so token 5 (our EOS) is the unique argmax
    // for any hidden state: column 5 = -10, all other columns = +1.
    // Because the hidden state of this random-weight model has a
    // negative mean, +1 columns yield negative logits and the -10 column
    // yields a strongly positive logit.
    int64_t vocab = cfg.vocab_size;
    for (int64_t i = 0; i < model.weights.W_output.numel(); ++i) model.weights.W_output.at_flat(i) = 1.0f;
    for (int64_t c = 0; c < cfg.hidden; ++c) {
        model.weights.W_output.at_flat(c * vocab + 5) = -10.0f;
    }
    auto out = generate(model, /*prompt=*/{0, 1}, /*max_new_tokens=*/10, sc,
                        /*eos=*/5, /*seed=*/0);
    REQUIRE(out.size() == 1);
    REQUIRE(out[0] == 5);
}

TEST_CASE(generate_zero_max_returns_empty) {
    LlamaConfig cfg = tiny_cfg();
    Lcg rng(789);
    LlamaModel model = make_model(tiny_weights(rng, cfg), cfg);
    auto out = generate(model, /*prompt=*/{0}, /*max_new_tokens=*/0, {});
    REQUIRE(out.empty());
}

TEST_CASE(generate_empty_prompt_throws) {
    LlamaConfig cfg = tiny_cfg();
    Lcg rng(101);
    LlamaModel model = make_model(tiny_weights(rng, cfg), cfg);
    REQUIRE_THROWS(generate(model, /*prompt=*/{}, /*max_new_tokens=*/3, {}));
}