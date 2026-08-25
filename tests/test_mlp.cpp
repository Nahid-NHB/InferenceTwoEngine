// tests/test_mlp.cpp
#include "test_helpers.hpp"
#include "tinyllm/mlp.hpp"

#include <cmath>

using namespace tinyllm;

TEST_CASE(mlp_zero_intermediate_weights_gives_zero) {
    // W_gate = W_up = 0 -> silu(0) * 0 = 0 -> matmul with anything = 0
    // W_down = 0 -> output = 0
    int64_t H = 3, I = 5;
    MlpWeights w;
    w.W_gate = Tensor({H, I}, DType::Float32); w.W_gate.fill(0.0f);
    w.W_up   = Tensor({H, I}, DType::Float32); w.W_up.fill(0.0f);
    w.W_down = Tensor({I, H}, DType::Float32); w.W_down.fill(0.0f);
    Tensor x({2, H}, DType::Float32); x.fill(2.0f);
    auto out = mlp_forward(x, w);
    REQUIRE(out.shape()[0] == 2);
    REQUIRE(out.shape()[1] == H);
    for (int64_t i = 0; i < out.numel(); ++i) {
        REQUIRE_NEAR(out.at_flat(i), 0.0f, 1e-6f);
    }
}

TEST_CASE(mlp_known_simple) {
    // hidden=2, intermediate=1. W_gate = [[1],[0]], W_up=[[0],[1]], W_down=[[1,1]]
    // x = [a, b]: gate=a, up=b. silu(a) ≈ a*sigmoid(a). h ≈ a*sigmoid(a)*b.
    // out = h * 1 = a*sigmoid(a)*b.
    int64_t H = 2, I = 1;
    MlpWeights w;
    w.W_gate = Tensor({H, I}, DType::Float32);
    w.W_gate.at_flat(0) = 1.0f; w.W_gate.at_flat(1) = 0.0f;
    w.W_up = Tensor({H, I}, DType::Float32);
    w.W_up.at_flat(0) = 0.0f; w.W_up.at_flat(1) = 1.0f;
    w.W_down = Tensor({I, H}, DType::Float32);
    w.W_down.fill(1.0f);

    Tensor x({1, H}, DType::Float32);
    x.at_flat(0) = 2.0f;
    x.at_flat(1) = 3.0f;
    auto out = mlp_forward(x, w);
    float silu2 = 2.0f / (1.0f + std::exp(-2.0f));
    REQUIRE_NEAR(out.at_flat(0), silu2 * 3.0f, 1e-5f);
    REQUIRE_NEAR(out.at_flat(1), silu2 * 3.0f, 1e-5f);
}

TEST_CASE(mlp_random_smoke) {
    int64_t H = 8, I = 12;
    MlpWeights w;
    w.W_gate = Tensor({H, I}, DType::Float32);
    w.W_up   = Tensor({H, I}, DType::Float32);
    w.W_down = Tensor({I, H}, DType::Float32);
    // Deterministic seed; just check shape and finiteness.
    uint64_t s = 99;
    for (int64_t i = 0; i < w.W_gate.numel(); ++i) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        w.W_gate.at_flat(i) = ((s >> 11) & 0xFFFF) / 65535.0f - 0.5f;
    }
    s = 123;
    for (int64_t i = 0; i < w.W_up.numel(); ++i) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        w.W_up.at_flat(i) = ((s >> 11) & 0xFFFF) / 65535.0f - 0.5f;
    }
    s = 456;
    for (int64_t i = 0; i < w.W_down.numel(); ++i) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        w.W_down.at_flat(i) = ((s >> 11) & 0xFFFF) / 65535.0f - 0.5f;
    }
    Tensor x({3, H}, DType::Float32);
    for (int64_t i = 0; i < x.numel(); ++i) x.at_flat(i) = static_cast<float>(i) * 0.1f;
    auto out = mlp_forward(x, w);
    REQUIRE(out.shape()[0] == 3);
    REQUIRE(out.shape()[1] == H);
    for (int64_t i = 0; i < out.numel(); ++i) {
        float v = out.at_flat(i);
        REQUIRE(!std::isnan(v));
        REQUIRE(!std::isinf(v));
    }
}