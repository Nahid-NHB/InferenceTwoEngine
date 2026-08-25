// tests/test_rope.cpp
#include "test_helpers.hpp"
#include "tinyllm/rope.hpp"

#include <cmath>

using namespace tinyllm;

TEST_CASE(rope_zero_position_identity) {
    // At position 0, all angles are 0, so cos=1, sin=0 -> x unchanged.
    Tensor x({1, 1, 4}, DType::Float32);
    x.at_flat(0) = 1.0f;
    x.at_flat(1) = 2.0f;
    x.at_flat(2) = 3.0f;
    x.at_flat(3) = 4.0f;
    Tensor orig({1, 1, 4}, DType::Float32);
    orig.at_flat(0) = 1.0f; orig.at_flat(1) = 2.0f;
    orig.at_flat(2) = 3.0f; orig.at_flat(3) = 4.0f;
    rope_inplace(x, /*start_pos=*/0);
    for (int64_t i = 0; i < 4; ++i) {
        REQUIRE_NEAR(x.at_flat(i), orig.at_flat(i), 1e-6f);
    }
}

TEST_CASE(rope_known_pair_rotation) {
    // head_dim=2, half=1, theta_base=10000.
    // For pair (0,1) at position 1: freq_0 = 10000^0 = 1, angle = 1.
    // cos(1) ≈ 0.5403, sin(1) ≈ 0.8415.
    Tensor x({1, 1, 2}, DType::Float32);
    x.at_flat(0) = 1.0f;  // row[i]
    x.at_flat(1) = 0.0f;  // row[i+half]
    rope_inplace(x, /*start_pos=*/1);
    float c = std::cos(1.0f), s = std::sin(1.0f);
    REQUIRE_NEAR(x.at_flat(0), 1.0f * c - 0.0f * s, 1e-5f);
    REQUIRE_NEAR(x.at_flat(1), 1.0f * s + 0.0f * c, 1e-5f);
}

TEST_CASE(rope_multiple_heads) {
    // 2 heads, head_dim=4, half=2. Check each head is rotated independently.
    Tensor x({1, 2, 4}, DType::Float32);
    // Head 0: [1, 0, 0, 1]
    x.at_flat(0) = 1.0f; x.at_flat(1) = 0.0f;
    x.at_flat(2) = 0.0f; x.at_flat(3) = 1.0f;
    // Head 1: [0, 1, 1, 0]
    x.at_flat(4) = 0.0f; x.at_flat(5) = 1.0f;
    x.at_flat(6) = 1.0f; x.at_flat(7) = 0.0f;
    rope_inplace(x, 0);  // pos 0 → identity
    for (int64_t i = 0; i < 8; ++i) {
        float orig = (i == 0 || i == 3 || i == 5 || i == 6) ? 1.0f : 0.0f;
        REQUIRE_NEAR(x.at_flat(i), orig, 1e-6f);
    }
}

TEST_CASE(rope_multiple_positions) {
    // 3 positions, 1 head, head_dim=4. Position 1 has non-trivial rotation.
    Tensor x({3, 1, 4}, DType::Float32);
    for (int64_t i = 0; i < x.numel(); ++i) x.at_flat(i) = static_cast<float>(i + 1);
    rope_inplace(x, 0);
    // Pos 0 = identity
    REQUIRE_NEAR(x.at_flat(0), 1.0f, 1e-5f);
    REQUIRE_NEAR(x.at_flat(1), 2.0f, 1e-5f);
    REQUIRE_NEAR(x.at_flat(2), 3.0f, 1e-5f);
    REQUIRE_NEAR(x.at_flat(3), 4.0f, 1e-5f);
    // Pos 1: rotated. For pair i=0 freq=10000^0=1 → angle=1.
//                  For pair i=1 freq=10000^(-2/4)=0.01 → angle=0.01.
    float c0 = std::cos(1.0f),       s0 = std::sin(1.0f);
    float c1 = std::cos(0.01f),      s1 = std::sin(0.01f);
    REQUIRE_NEAR(x.at_flat(4), 5.0f * c0 - 7.0f * s0, 1e-5f);
    REQUIRE_NEAR(x.at_flat(5), 6.0f * c1 - 8.0f * s1, 1e-5f);
    REQUIRE_NEAR(x.at_flat(6), 5.0f * s0 + 7.0f * c0, 1e-5f);
    REQUIRE_NEAR(x.at_flat(7), 6.0f * s1 + 8.0f * c1, 1e-5f);
}

TEST_CASE(rope_start_pos_offset) {
    // start_pos=2 → position 0 of input is at global position 2.
    Tensor x({1, 1, 2}, DType::Float32);
    x.at_flat(0) = 1.0f; x.at_flat(1) = 0.0f;
    rope_inplace(x, /*start_pos=*/2);
    float c = std::cos(2.0f), s = std::sin(2.0f);
    REQUIRE_NEAR(x.at_flat(0), 1.0f * c - 0.0f * s, 1e-5f);
    REQUIRE_NEAR(x.at_flat(1), 1.0f * s + 0.0f * c, 1e-5f);
}

TEST_CASE(rope_preserves_norm_per_pair) {
    // RoPE is a rotation, so for each pair (x_lo, x_hi) → (new_lo, new_hi),
    // the L2 norm is preserved: x_lo^2 + x_hi^2 == new_lo^2 + new_hi^2.
    Tensor x({1, 1, 4}, DType::Float32);
    x.at_flat(0) = 1.0f; x.at_flat(1) = 2.0f;
    x.at_flat(2) = 3.0f; x.at_flat(3) = 4.0f;
    float n0 = x.at_flat(0)*x.at_flat(0) + x.at_flat(2)*x.at_flat(2);  // pair (0,2)
    float n1 = x.at_flat(1)*x.at_flat(1) + x.at_flat(3)*x.at_flat(3);  // pair (1,3)
    rope_inplace(x, /*start_pos=*/5);
    float n0p = x.at_flat(0)*x.at_flat(0) + x.at_flat(2)*x.at_flat(2);
    float n1p = x.at_flat(1)*x.at_flat(1) + x.at_flat(3)*x.at_flat(3);
    REQUIRE_NEAR(n0p, n0, 1e-5f);
    REQUIRE_NEAR(n1p, n1, 1e-5f);
}

TEST_CASE(rope_precompute_tables) {
    // Tables are independent of any input tensor.
    auto t = precompute_rope_tables(/*seq_len=*/4, /*head_dim=*/4, /*start_pos=*/0);
    REQUIRE(t.cos.shape()[0] == 4);
    REQUIRE(t.cos.shape()[1] == 2);
    REQUIRE(t.sin.shape()[0] == 4);
    REQUIRE(t.sin.shape()[1] == 2);
    // Pos 0: cos = 1, sin = 0
    REQUIRE_NEAR(t.cos.at_flat(0), 1.0f, 1e-6f);
    REQUIRE_NEAR(t.sin.at_flat(0), 0.0f, 1e-6f);
    REQUIRE_NEAR(t.cos.at_flat(1), 1.0f, 1e-6f);
    REQUIRE_NEAR(t.sin.at_flat(1), 0.0f, 1e-6f);
}

TEST_CASE(rope_rejects_odd_head_dim) {
    Tensor x({1, 1, 3}, DType::Float32); x.fill(0.0f);
    REQUIRE_THROWS(rope_inplace(x, 0));
}
