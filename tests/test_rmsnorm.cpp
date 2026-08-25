// tests/test_rmsnorm.cpp
#include "test_helpers.hpp"
#include "tinyllm/rmsnorm.hpp"

#include <cmath>
#include <vector>

using namespace tinyllm;

TEST_CASE(rmsnorm_zero_input) {
    // Zero input → RMS = sqrt(eps), output = 0 * gamma / sqrt(eps) = 0.
    Tensor x({1, 4}, DType::Float32);
    x.fill(0.0f);
    Tensor g({4}, DType::Float32);
    g.fill(1.0f);
    auto y = rmsnorm(x, g, 1e-5f);
    REQUIRE(y.shape().size() == 2);
    REQUIRE(y.shape()[0] == 1);
    REQUIRE(y.shape()[1] == 4);
    for (int64_t i = 0; i < y.numel(); ++i) {
        REQUIRE_NEAR(y.at_flat(i), 0.0f, 1e-6f);
    }
}

TEST_CASE(rmsnorm_known_values) {
    // d=2, gamma=[1,1], eps=0.
    //   x=[3,4] -> mean_sq = (9+16)/2 = 12.5, rms = sqrt(12.5)
    //   y = [3,4] / sqrt(12.5)
    Tensor x({2}, DType::Float32);
    x.at_flat(0) = 3.0f; x.at_flat(1) = 4.0f;
    Tensor g({2}, DType::Float32);
    g.fill(1.0f);
    auto y = rmsnorm(x, g, 0.0f);
    float rms = std::sqrt(12.5f);
    REQUIRE_NEAR(y.at_flat(0), 3.0f / rms, 1e-5f);
    REQUIRE_NEAR(y.at_flat(1), 4.0f / rms, 1e-5f);
}

TEST_CASE(rmsnorm_with_gamma) {
    // Multiply by gamma per-channel.
    Tensor x({1, 3}, DType::Float32);
    x.at_flat(0) = 1.0f; x.at_flat(1) = 2.0f; x.at_flat(2) = 2.0f;
    Tensor g({3}, DType::Float32);
    g.at_flat(0) = 2.0f; g.at_flat(1) = 0.0f; g.at_flat(2) = 1.0f;
    auto y = rmsnorm(x, g, 0.0f);
    // mean_sq = (1+4+4)/3 = 3, rms = sqrt(3)
    float rms = std::sqrt(3.0f);
    REQUIRE_NEAR(y.at_flat(0), 2.0f * (1.0f / rms), 1e-5f);
    REQUIRE_NEAR(y.at_flat(1), 0.0f, 1e-6f);
    REQUIRE_NEAR(y.at_flat(2), 1.0f * (2.0f / rms), 1e-5f);
}

TEST_CASE(rmsnorm_2d_input) {
    // Two rows, each normalized independently.
    Tensor x({2, 4}, DType::Float32);
    x.at_flat(0) = 1.0f; x.at_flat(1) = 0.0f; x.at_flat(2) = 0.0f; x.at_flat(3) = 0.0f;
    x.at_flat(4) = 0.0f; x.at_flat(5) = 0.0f; x.at_flat(6) = 3.0f; x.at_flat(7) = 4.0f;
    Tensor g({4}, DType::Float32);
    g.fill(1.0f);
    auto y = rmsnorm(x, g, 0.0f);
    // Row 0: only first element non-zero, RMS = 0.5 → out = [2,0,0,0]
    REQUIRE_NEAR(y.at_flat(0), 2.0f, 1e-5f);
    REQUIRE_NEAR(y.at_flat(1), 0.0f, 1e-6f);
    REQUIRE_NEAR(y.at_flat(2), 0.0f, 1e-6f);
    REQUIRE_NEAR(y.at_flat(3), 0.0f, 1e-6f);
    // Row 1: [0,0,3,4] → mean_sq = (0+0+9+16)/4 = 6.25, rms = 2.5.
    float rms = std::sqrt(6.25f);
    REQUIRE_NEAR(y.at_flat(4), 0.0f, 1e-6f);
    REQUIRE_NEAR(y.at_flat(5), 0.0f, 1e-6f);
    REQUIRE_NEAR(y.at_flat(6), 3.0f / rms, 1e-5f);
    REQUIRE_NEAR(y.at_flat(7), 4.0f / rms, 1e-5f);
}

TEST_CASE(rmsnorm_3d_input) {
    // [..., d] interpretation works for arbitrary leading dims.
    Tensor x({2, 3, 2}, DType::Float32);
    for (int64_t i = 0; i < x.numel(); ++i) x.at_flat(i) = static_cast<float>(i + 1);
    Tensor g({2}, DType::Float32);
    g.fill(1.0f);
    auto y = rmsnorm(x, g, 0.0f);
    REQUIRE(y.shape()[0] == 2);
    REQUIRE(y.shape()[1] == 3);
    REQUIRE(y.shape()[2] == 2);
    REQUIRE(y.numel() == 12);
}

TEST_CASE(rmsnorm_rejects_wrong_last_dim) {
    Tensor x({4}, DType::Float32); x.fill(1.0f);
    Tensor g({3}, DType::Float32); g.fill(1.0f);
    REQUIRE_THROWS(rmsnorm(x, g));
}

TEST_CASE(rmsnorm_rejects_non_1d_gamma) {
    Tensor x({4}, DType::Float32); x.fill(1.0f);
    Tensor g({2, 2}, DType::Float32); g.fill(1.0f);
    REQUIRE_THROWS(rmsnorm(x, g));
}
