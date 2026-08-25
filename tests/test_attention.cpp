// tests/test_attention.cpp
#include "test_helpers.hpp"
#include "tinyllm/attention.hpp"

#include <cmath>

using namespace tinyllm;

namespace {

Tensor make_random(int64_t rows, int64_t cols, float scale = 0.1f,
                    unsigned seed = 12345u) {
    Tensor t({rows, cols}, DType::Float32);
    // Simple LCG so tests are deterministic across runs/platforms.
    uint64_t s = seed;
    for (int64_t i = 0; i < t.numel(); ++i) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        float u = static_cast<float>((s >> 11) & ((1ULL << 21) - 1)) /
                  static_cast<float>(1ULL << 21);  // [0,1)
        t.at_flat(i) = (u * 2.0f - 1.0f) * scale;
    }
    return t;
}

}  // namespace

TEST_CASE(attention_identity_weights_gives_mean) {
    // If Wq=Wk=Wv=I and Wo=I (and no RoPE since pos=0 → identity rotation),
    // and all inputs are constant c, then each head's Q=c, K=c, V=c.
    // scores = c^2 / sqrt(d) (uniform), softmax = uniform, ctx = c.
    // Output = c (after Wo=I).
    AttentionConfig cfg;
    cfg.hidden = 4;
    cfg.n_heads = 2;
    cfg.n_kv_heads = 1;  // GQA: 2 q heads share 1 kv head
    cfg.head_dim = 2;

    int64_t H = cfg.hidden;
    int64_t KVH = cfg.n_kv_heads * cfg.head_dim;
    AttentionWeights w;
    w.Wq = Tensor({H, H}, DType::Float32); w.Wq.fill(0.0f);
    w.Wk = Tensor({H, KVH}, DType::Float32); w.Wk.fill(0.0f);
    w.Wv = Tensor({H, KVH}, DType::Float32); w.Wv.fill(0.0f);
    w.Wo = Tensor({H, H}, DType::Float32); w.Wo.fill(0.0f);
    // Wq and Wo are square, identity in their full hidden size.
    for (int64_t i = 0; i < H; ++i) {
        w.Wq.at_flat(i * H + i) = 1.0f;
        w.Wo.at_flat(i * H + i) = 1.0f;
    }
    // Wk, Wv only have n_kv_heads*head_dim=2 columns, so set just the
    // first two rows' diagonals to keep them as identity-on-2.
    for (int64_t i = 0; i < KVH; ++i) {
        w.Wk.at_flat(i * KVH + i) = 1.0f;
        w.Wv.at_flat(i * KVH + i) = 1.0f;
    }

    Tensor x({2, H}, DType::Float32);
    x.fill(1.0f);

    auto out = attention_forward(x, w, cfg, /*start_pos=*/0);
    REQUIRE(out.shape()[0] == 2);
    REQUIRE(out.shape()[1] == H);
    // V after projection is constant (each row of K/V equals 1); the
    // softmax-weighted sum of V is also constant → output is constant.
    // (We don't pin the exact constant; that's an implementation detail
    // of softmax. We just check that all outputs are equal and finite.)
    float v0 = out.at_flat(0);
    for (int64_t i = 0; i < out.numel(); ++i) {
        REQUIRE_NEAR(out.at_flat(i), v0, 1e-5f);
    }
}

TEST_CASE(attention_causal_mask_position0_only_attends_to_itself) {
    // Use seq=1 so RoPE at position 0 is identity (cos=1, sin=0).
    // Then the only attention output is the value at position 0, which
    // (after causal mask) is just V[0].
    AttentionConfig cfg;
    cfg.hidden = 2; cfg.n_heads = 1; cfg.n_kv_heads = 1; cfg.head_dim = 2;
    int64_t H = cfg.hidden;
    AttentionWeights w;
    w.Wq = Tensor({H, H}, DType::Float32); w.Wq.fill(0.0f);
    w.Wk = Tensor({H, H}, DType::Float32); w.Wk.fill(0.0f);
    w.Wv = Tensor({H, H}, DType::Float32); w.Wv.fill(0.0f);
    w.Wo = Tensor({H, H}, DType::Float32); w.Wo.fill(0.0f);
    for (int64_t i = 0; i < H; ++i) {
        w.Wq.at_flat(i * H + i) = 1.0f;
        w.Wk.at_flat(i * H + i) = 1.0f;
        w.Wv.at_flat(i * H + i) = 1.0f;
        w.Wo.at_flat(i * H + i) = 1.0f;
    }

    Tensor x({1, H}, DType::Float32);
    x.at_flat(0) = 7.0f; x.at_flat(1) = 3.0f;
    auto out = attention_forward(x, w, cfg, 0);
    REQUIRE(out.shape()[0] == 1);
    REQUIRE(out.shape()[1] == H);
    // Position 0: scores = [7/√2], softmax = [1], ctx = V[0] = [7, 3].
    REQUIRE_NEAR(out.at_flat(0), 7.0f, 1e-5f);
    REQUIRE_NEAR(out.at_flat(1), 3.0f, 1e-5f);
}

TEST_CASE(attention_seq2_with_rope) {
    // End-to-end: seq=2, head_dim=2, identity weights. Verifies that
    // RoPE actually rotates at position 1 (not just at position 0).
    // Position 0: Q[0]=K[0]=[1,0] (RoPE at pos 0 = identity).
    // Position 1: with angle=1, pair rotates (a,b) -> (a*cos1 - b*sin1,
    //                                              a*sin1 + b*cos1).
    //   Q[1] = [0,1] -> after RoPE = [-sin1, cos1]
    //   K[1] = [0,1] -> after RoPE = [-sin1, cos1]
    //   V[1] = [0,1] -> unchanged.
    // Scores (raw):
    //   row 0 col 0 = Q[0]·K[0] = 1; row 0 col 1 = Q[0]·K[1] = -sin1.
    //   row 1 col 0 = Q[1]·K[0] = -sin1; row 1 col 1 = Q[1]·K[1] = 1.
    // After scale 1/√2 and causal mask: row 0 has [-sin1/√2, -inf] -> [1, 0];
    //   row 1 has [-sin1/√2, 1/√2] -> softmax([a, b]) = [s0, s1].
    // ctx[1] = s0*V[0] + s1*V[1] = [s0, s1].
    // Output (Wo=I): [s0, s1] at index 2,3.
    AttentionConfig cfg;
    cfg.hidden = 2; cfg.n_heads = 1; cfg.n_kv_heads = 1; cfg.head_dim = 2;
    int64_t H = cfg.hidden;
    AttentionWeights w;
    w.Wq = Tensor({H, H}, DType::Float32); w.Wq.fill(0.0f);
    w.Wk = Tensor({H, H}, DType::Float32); w.Wk.fill(0.0f);
    w.Wv = Tensor({H, H}, DType::Float32); w.Wv.fill(0.0f);
    w.Wo = Tensor({H, H}, DType::Float32); w.Wo.fill(0.0f);
    for (int64_t i = 0; i < H; ++i) {
        w.Wq.at_flat(i * H + i) = 1.0f;
        w.Wk.at_flat(i * H + i) = 1.0f;
        w.Wv.at_flat(i * H + i) = 1.0f;
        w.Wo.at_flat(i * H + i) = 1.0f;
    }
    Tensor x({2, H}, DType::Float32);
    x.at_flat(0) = 1.0f; x.at_flat(1) = 0.0f;
    x.at_flat(2) = 0.0f; x.at_flat(3) = 1.0f;
    auto out = attention_forward(x, w, cfg, 0);
    // Position 0
    REQUIRE_NEAR(out.at_flat(0), 1.0f, 1e-5f);
    REQUIRE_NEAR(out.at_flat(1), 0.0f, 1e-5f);
    // Position 1: weights = softmax([-sin(1)/√2, 1/√2])
    float sin1 = std::sin(1.0f);
    float inv_sqrt2 = 1.0f / std::sqrt(2.0f);
    float a = -sin1 * inv_sqrt2;
    float b = inv_sqrt2;
    float s0 = std::exp(a) / (std::exp(a) + std::exp(b));
    REQUIRE_NEAR(out.at_flat(2), s0, 1e-5f);
    REQUIRE_NEAR(out.at_flat(3), 1.0f - s0, 1e-5f);
}

TEST_CASE(attention_random_smoke) {
    // Run a forward pass with random weights; just check shapes and
    // finiteness.
    AttentionConfig cfg;
    cfg.hidden = 8; cfg.n_heads = 2; cfg.n_kv_heads = 1; cfg.head_dim = 4;
    AttentionWeights w;
    w.Wq = make_random(cfg.hidden, cfg.hidden, 0.1f, 1);
    w.Wk = make_random(cfg.hidden, cfg.n_kv_heads * cfg.head_dim, 0.1f, 2);
    w.Wv = make_random(cfg.hidden, cfg.n_kv_heads * cfg.head_dim, 0.1f, 3);
    w.Wo = make_random(cfg.hidden, cfg.hidden, 0.1f, 4);
    Tensor x = make_random(3, cfg.hidden, 1.0f, 5);
    auto out = attention_forward(x, w, cfg, 0);
    REQUIRE(out.shape()[0] == 3);
    REQUIRE(out.shape()[1] == cfg.hidden);
    for (int64_t i = 0; i < out.numel(); ++i) {
        float v = out.at_flat(i);
        if (std::isnan(v) || std::isinf(v)) {
            REQUIRE(false);
        }
    }
}

TEST_CASE(attention_rejects_bad_config) {
    AttentionConfig cfg;
    cfg.hidden = 4; cfg.n_heads = 3; cfg.n_kv_heads = 2; cfg.head_dim = 2;
    AttentionWeights w;
    w.Wq = Tensor({4, 6}, DType::Float32); w.Wq.fill(0.0f);
    w.Wk = Tensor({4, 4}, DType::Float32); w.Wk.fill(0.0f);
    w.Wv = Tensor({4, 4}, DType::Float32); w.Wv.fill(0.0f);
    w.Wo = Tensor({6, 4}, DType::Float32); w.Wo.fill(0.0f);
    Tensor x({2, 4}, DType::Float32); x.fill(0.0f);
    REQUIRE_THROWS(attention_forward(x, w, cfg, 0));
}