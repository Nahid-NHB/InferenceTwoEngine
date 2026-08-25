// tests/test_tensor.cpp
// -----------------------------------------------------------------------------
// Unit tests for the Phase 1 tensor library.
// -----------------------------------------------------------------------------
#include "test_helpers.hpp"
#include "tinyllm/tensor.hpp"
#include "tinyllm/matmul.hpp"

#include <cmath>
#include <numeric>
#include <vector>

using namespace tinyllm;

// -----------------------------------------------------------------------------
// Construction & metadata
// -----------------------------------------------------------------------------
TEST_CASE(tensor_basic_construction) {
    Tensor t({2, 3}, DType::Float32);
    REQUIRE(t.ndim() == 2);
    REQUIRE(t.shape()[0] == 2);
    REQUIRE(t.shape()[1] == 3);
    REQUIRE(t.numel() == 6);
    REQUIRE(t.dtype() == DType::Float32);
    REQUIRE(t.is_contiguous());
}

TEST_CASE(tensor_init_list_shape) {
    Tensor t({2, 3, 4}, DType::Float32);
    REQUIRE(t.ndim() == 3);
    REQUIRE(t.numel() == 24);
    REQUIRE(t.strides()[0] == 12);
    REQUIRE(t.strides()[1] == 4);
    REQUIRE(t.strides()[2] == 1);
}

TEST_CASE(tensor_init_from_buffer) {
    std::vector<float> data{1, 2, 3, 4, 5, 6};
    Tensor t({2, 3}, DType::Float32, data.data(), data.size() * sizeof(float));
    REQUIRE(t.at_flat(0) == 1.0f);
    REQUIRE(t.at_flat(5) == 6.0f);
}

TEST_CASE(empty_tensor_1_element) {
    Tensor t({});
    REQUIRE(t.ndim() == 0);
    REQUIRE(t.numel() == 1);
}

// -----------------------------------------------------------------------------
// fill / scalar ops
// -----------------------------------------------------------------------------
TEST_CASE(tensor_fill) {
    Tensor t({2, 3}, DType::Float32);
    t.fill(3.5f);
    for (int64_t i = 0; i < t.numel(); ++i) REQUIRE(t.at_flat(i) == 3.5f);
}

TEST_CASE(add_scalar) {
    Tensor t({2, 2}, DType::Float32);
    t.fill(2.0f);
    auto r = ops::add_scalar(t, 1.0f);
    for (int64_t i = 0; i < r.numel(); ++i) REQUIRE(r.at_flat(i) == 3.0f);
}

TEST_CASE(mul_scalar) {
    Tensor t({3}, DType::Float32);
    auto d = std::vector<float>{1, 2, 3};
    Tensor t2({3}, DType::Float32, d.data(), d.size() * sizeof(float));
    auto r = ops::mul_scalar(t2, 2.0f);
    REQUIRE(r.at_flat(0) == 2.0f);
    REQUIRE(r.at_flat(1) == 4.0f);
    REQUIRE(r.at_flat(2) == 6.0f);
}

// -----------------------------------------------------------------------------
// Element-wise ops + broadcasting
// -----------------------------------------------------------------------------
TEST_CASE(elementwise_add_same_shape) {
    auto d1 = std::vector<float>{1, 2, 3, 4};
    auto d2 = std::vector<float>{10, 20, 30, 40};
    Tensor a({4}, DType::Float32, d1.data(), d1.size() * sizeof(float));
    Tensor b({4}, DType::Float32, d2.data(), d2.size() * sizeof(float));
    auto c = ops::add(a, b);
    REQUIRE(c.at_flat(0) == 11.0f);
    REQUIRE(c.at_flat(3) == 44.0f);
}

TEST_CASE(elementwise_with_broadcasting) {
    // shape [2,3] + shape [3] → [2,3]
    auto d1 = std::vector<float>{1, 2, 3, 4, 5, 6};
    auto d2 = std::vector<float>{10, 20, 30};
    Tensor a({2, 3}, DType::Float32, d1.data(), d1.size() * sizeof(float));
    Tensor b({3},    DType::Float32, d2.data(), d2.size() * sizeof(float));
    auto c = ops::add(a, b);
    REQUIRE(c.shape()[0] == 2);
    REQUIRE(c.shape()[1] == 3);
    // First row: [11, 22, 33]; second row: [14, 25, 36]
    REQUIRE(c.at_flat(0) == 11.0f);
    REQUIRE(c.at_flat(1) == 22.0f);
    REQUIRE(c.at_flat(2) == 33.0f);
    REQUIRE(c.at_flat(3) == 14.0f);
    REQUIRE(c.at_flat(4) == 25.0f);
    REQUIRE(c.at_flat(5) == 36.0f);
}

TEST_CASE(elementwise_broadcast_row_vec) {
    // [2,3] + [2,1]
    auto d1 = std::vector<float>{1, 2, 3, 4, 5, 6};
    auto d2 = std::vector<float>{10, 100};
    Tensor a({2, 3}, DType::Float32, d1.data(), d1.size() * sizeof(float));
    Tensor b({2, 1}, DType::Float32, d2.data(), d2.size() * sizeof(float));
    auto c = ops::add(a, b);
    REQUIRE(c.at_flat(0) == 11.0f);
    REQUIRE(c.at_flat(1) == 12.0f);
    REQUIRE(c.at_flat(2) == 13.0f);
    REQUIRE(c.at_flat(3) == 104.0f);
    REQUIRE(c.at_flat(4) == 105.0f);
    REQUIRE(c.at_flat(5) == 106.0f);
}

// -----------------------------------------------------------------------------
// Reductions
// -----------------------------------------------------------------------------
TEST_CASE(sum_all_elements) {
    auto d = std::vector<float>{1, 2, 3, 4};
    Tensor t({4}, DType::Float32, d.data(), d.size() * sizeof(float));
    auto s = ops::sum(t);
    REQUIRE(s.ndim() == 0);
    REQUIRE(s.numel() == 1);
    REQUIRE_NEAR(s.at_flat(0), 10.0f, 1e-6);
}

TEST_CASE(sum_axis_keepdims) {
    auto d = std::vector<float>{1, 2, 3, 4, 5, 6};
    Tensor t({2, 3}, DType::Float32, d.data(), d.size() * sizeof(float));
    auto s = ops::sum(t, /*axis=*/0, /*keepdims=*/true);
    REQUIRE(s.shape()[0] == 1);
    REQUIRE(s.shape()[1] == 3);
    REQUIRE_NEAR(s.at_flat(0), 5.0f, 1e-6);
    REQUIRE_NEAR(s.at_flat(1), 7.0f, 1e-6);
    REQUIRE_NEAR(s.at_flat(2), 9.0f, 1e-6);
}

TEST_CASE(mean_axis) {
    auto d = std::vector<float>{1, 2, 3, 4, 5, 6};
    Tensor t({2, 3}, DType::Float32, d.data(), d.size() * sizeof(float));
    auto mv = ops::mean(t, /*axis=*/1);
    REQUIRE(mv.shape().size() == 1);
    REQUIRE(mv.shape()[0] == 2);
    REQUIRE_NEAR(mv.at_flat(0), 2.0f, 1e-6);
    REQUIRE_NEAR(mv.at_flat(1), 5.0f, 1e-6);
}

// -----------------------------------------------------------------------------
// softmax
// -----------------------------------------------------------------------------
TEST_CASE(softmax_rows_sum_to_one) {
    auto d = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    Tensor t({2, 4}, DType::Float32, d.data(), d.size() * sizeof(float));
    auto s = ops::softmax(t, /*axis=*/1);
    REQUIRE(s.shape() == t.shape());
    for (int row = 0; row < 2; ++row) {
        double sum = 0.0;
        for (int col = 0; col < 4; ++col) {
            float v = s.at_flat(row * 4 + col);
            REQUIRE(v > 0.0f);
            sum += static_cast<double>(v);
        }
        REQUIRE_NEAR(sum, 1.0, 1e-5);
    }
}

TEST_CASE(softmax_uniform_input_is_uniform) {
    auto d = std::vector<float>{1.0f, 1.0f, 1.0f, 1.0f};
    Tensor t({4}, DType::Float32, d.data(), d.size() * sizeof(float));
    auto s = ops::softmax(t);
    for (int i = 0; i < 4; ++i) REQUIRE_NEAR(s.at_flat(i), 0.25f, 1e-6);
}

TEST_CASE(softmax_numerical_stability) {
    // Large values would overflow without max-subtraction.
    auto d = std::vector<float>{1000.0f, 1001.0f, 1002.0f};
    Tensor t({3}, DType::Float32, d.data(), d.size() * sizeof(float));
    auto s = ops::softmax(t);
    REQUIRE(std::isfinite(s.at_flat(0)));
    double sum = s.at_flat(0) + s.at_flat(1) + s.at_flat(2);
    REQUIRE_NEAR(sum, 1.0, 1e-5);
}

// -----------------------------------------------------------------------------
// Matmul (naive)
// -----------------------------------------------------------------------------
TEST_CASE(matmul_identity) {
    auto d = std::vector<float>{1, 2, 3, 4, 5, 6};
    Tensor a({2, 3}, DType::Float32, d.data(), d.size() * sizeof(float));
    std::vector<float> eye{1, 0, 0, 0, 1, 0, 0, 0, 1};
    Tensor id({3, 3}, DType::Float32, eye.data(), eye.size() * sizeof(float));
    auto r = ops::matmul(a, id);
    REQUIRE(r.shape()[0] == 2);
    REQUIRE(r.shape()[1] == 3);
    REQUIRE_NEAR(r.at_flat(0), 1.0f, 1e-5);
    REQUIRE_NEAR(r.at_flat(1), 2.0f, 1e-5);
    REQUIRE_NEAR(r.at_flat(2), 3.0f, 1e-5);
    REQUIRE_NEAR(r.at_flat(3), 4.0f, 1e-5);
}

TEST_CASE(matmul_2x3_times_3x2) {
    auto da = std::vector<float>{1, 2, 3, 4, 5, 6};          // [[1,2,3],[4,5,6]]
    auto db = std::vector<float>{7, 8, 9, 10, 11, 12};      // [[7,8],[9,10],[11,12]]
    Tensor a({2, 3}, DType::Float32, da.data(), da.size() * sizeof(float));
    Tensor b({3, 2}, DType::Float32, db.data(), db.size() * sizeof(float));
    auto r = ops::matmul(a, b);
    // Expected:
    //   [1*7+2*9+3*11,  1*8+2*10+3*12] = [58, 64]
    //   [4*7+5*9+6*11,  4*8+5*10+6*12] = [139, 154]
    REQUIRE(r.shape()[0] == 2);
    REQUIRE(r.shape()[1] == 2);
    REQUIRE_NEAR(r.at_flat(0), 58.0f,  1e-4);
    REQUIRE_NEAR(r.at_flat(1), 64.0f,  1e-4);
    REQUIRE_NEAR(r.at_flat(2), 139.0f, 1e-4);
    REQUIRE_NEAR(r.at_flat(3), 154.0f, 1e-4);
}

// -----------------------------------------------------------------------------
// Views: reshape, transpose, contiguous
// -----------------------------------------------------------------------------
TEST_CASE(reshape_preserves_values) {
    auto d = std::vector<float>{1, 2, 3, 4, 5, 6};
    Tensor t({6}, DType::Float32, d.data(), d.size() * sizeof(float));
    auto r = t.reshape({2, 3});
    REQUIRE(r.shape()[0] == 2);
    REQUIRE(r.shape()[1] == 3);
    REQUIRE(r.at_flat(0) == 1.0f);
    REQUIRE(r.at_flat(3) == 4.0f);
    REQUIRE(r.at_flat(5) == 6.0f);
}

TEST_CASE(transpose_view_zero_copy) {
    auto d = std::vector<float>{1, 2, 3, 4, 5, 6};
    Tensor t({2, 3}, DType::Float32, d.data(), d.size() * sizeof(float));
    auto tt = t.transpose();
    REQUIRE(tt.shape()[0] == 3);
    REQUIRE(tt.shape()[1] == 2);
    REQUIRE(!tt.is_contiguous());
    REQUIRE(tt.storage().get() == t.storage().get());  // same underlying buffer
    REQUIRE(tt.at({0, 0}) == 1.0f);  // tt[0,0] = t[0,0]
    REQUIRE(tt.at({0, 1}) == 4.0f);  // tt[0,1] = t[1,0]
    REQUIRE(tt.at({1, 0}) == 2.0f);  // tt[1,0] = t[0,1]
    REQUIRE(tt.at({2, 1}) == 6.0f);  // tt[2,1] = t[1,2]
}

TEST_CASE(contiguous_after_transpose) {
    auto d = std::vector<float>{1, 2, 3, 4, 5, 6};
    Tensor t({2, 3}, DType::Float32, d.data(), d.size() * sizeof(float));
    auto c = t.transpose().contiguous();
    REQUIRE(c.is_contiguous());
    // After transpose and copy, the new layout is the original transposed:
    //   [[1,4],[2,5],[3,6]]
    REQUIRE_NEAR(c.at_flat(0), 1.0f, 1e-6);
    REQUIRE_NEAR(c.at_flat(1), 4.0f, 1e-6);
    REQUIRE_NEAR(c.at_flat(2), 2.0f, 1e-6);
    REQUIRE_NEAR(c.at_flat(3), 5.0f, 1e-6);
    REQUIRE_NEAR(c.at_flat(4), 3.0f, 1e-6);
    REQUIRE_NEAR(c.at_flat(5), 6.0f, 1e-6);
}

TEST_CASE(squeeze_and_unsqueeze) {
    auto d = std::vector<float>{1, 2, 3};
    Tensor t({1, 3}, DType::Float32, d.data(), d.size() * sizeof(float));
    auto s = t.squeeze();
    REQUIRE(s.shape().size() == 1);
    REQUIRE(s.shape()[0] == 3);
    auto u = s.unsqueeze(0);
    REQUIRE(u.shape().size() == 2);
    REQUIRE(u.shape()[0] == 1);
    REQUIRE(u.shape()[1] == 3);
}

// -----------------------------------------------------------------------------
// Storage sharing (refcount sanity check)
// -----------------------------------------------------------------------------
TEST_CASE(storage_shared_across_views) {
    auto d = std::vector<float>{1, 2, 3, 4};
    Tensor t({2, 2}, DType::Float32, d.data(), d.size() * sizeof(float));
    REQUIRE(t.storage().use_count() == 1);
    auto r = t.reshape({4});
    REQUIRE(r.storage().get() == t.storage().get());
    REQUIRE(t.storage().use_count() == 2);
}