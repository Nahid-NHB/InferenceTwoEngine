// tests/test_matmul.cpp
// -----------------------------------------------------------------------------
// Correctness tests for the Phase 2 matmul variants.
//
// Each variant must produce the same result as the naive reference (modulo
// floating-point reorder). We also sanity-check:
//   - shape is correct
//   - identity multiplication is exact
//   - matmul vs hand-computed reference values match within 1e-3
// -----------------------------------------------------------------------------
#include "test_helpers.hpp"
#include "tinyllm/matmul.hpp"
#include "tinyllm/tensor.hpp"

#include <random>
#include <vector>

using namespace tinyllm;

static Tensor make_random(int64_t rows, int64_t cols, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    std::vector<float> v(static_cast<std::size_t>(rows * cols));
    for (auto& x : v) x = d(rng);
    return Tensor({rows, cols}, DType::Float32, v.data(), v.size() * sizeof(float));
}

// Compare two tensors element-wise within tol. Different loop orderings
// accumulate rounding error differently, so we use a generous 1e-3.
static void require_close(const Tensor& a, const Tensor& b, float tol) {
    REQUIRE(a.shape() == b.shape());
    REQUIRE(a.dtype() == b.dtype());
    const float* pa = a.data_float();
    const float* pb = b.data_float();
    for (int64_t i = 0; i < a.numel(); ++i) {
        REQUIRE_NEAR(pa[i], pb[i], tol);
    }
}

TEST_CASE(matmul_naive_correctness) {
    auto A = make_random(7,  5, 1);
    auto B = make_random(5, 11, 2);
    auto C = ops::matmul_naive(A, B);
    REQUIRE(C.shape()[0] == 7);
    REQUIRE(C.shape()[1] == 11);
    // Compare against hand-computed reference for a tiny fixed case.
    std::vector<float> ra{1, 2, 3,
                          4, 5, 6};
    std::vector<float> rb{ 7,  8,
                           9, 10,
                          11, 12};
    Tensor ta({2, 3}, DType::Float32, ra.data(), ra.size() * sizeof(float));
    Tensor tb({3, 2}, DType::Float32, rb.data(), rb.size() * sizeof(float));
    auto tc = ops::matmul_naive(ta, tb);
    REQUIRE_NEAR(tc.at({0, 0}), 58.0f,  1e-3);
    REQUIRE_NEAR(tc.at({0, 1}), 64.0f,  1e-3);
    REQUIRE_NEAR(tc.at({1, 0}), 139.0f, 1e-3);
    REQUIRE_NEAR(tc.at({1, 1}), 154.0f, 1e-3);
}

TEST_CASE(variants_agree_on_random) {
    auto A = make_random(64, 64, 7);
    auto B = make_random(64, 64, 8);
    auto Cn  = ops::matmul_naive  (A, B);
    auto Cb  = ops::matmul_blocked(A, B);
    auto Ca  = ops::matmul_avx2   (A, B);
    auto Ct  = ops::matmul_threaded(A, B);
    require_close(Cn, Cb, 1e-3);
    require_close(Cn, Ca, 1e-3);
    require_close(Cn, Ct, 1e-3);
}

TEST_CASE(variants_agree_large) {
    auto A = make_random(256, 256, 11);
    auto B = make_random(256, 256, 12);
    auto Cn = ops::matmul_naive  (A, B);
    auto Cb = ops::matmul_blocked(A, B);
    auto Ca = ops::matmul_avx2   (A, B);
    auto Ct = ops::matmul_threaded(A, B);
    require_close(Cn, Cb, 1e-3);
    require_close(Cn, Ca, 1e-3);
    require_close(Cn, Ct, 1e-3);
}

TEST_CASE(variants_agree_non_square) {
    auto A = make_random(37, 53, 13);
    auto B = make_random(53, 91, 14);
    auto Cn = ops::matmul_naive  (A, B);
    auto Ca = ops::matmul_avx2   (A, B);
    auto Ct = ops::matmul_threaded(A, B);
    require_close(Cn, Ca, 1e-3);
    require_close(Cn, Ct, 1e-3);
}

TEST_CASE(variants_agree_tail_sizes) {
    // Sizes not divisible by the AVX2 micro-tile (6 rows, 16 cols).
    // The fallback path inside mm_avx2_kernel must still produce correct results.
    auto A = make_random(70, 130, 21);
    auto B = make_random(130, 50, 22);
    auto Cn = ops::matmul_naive(A, B);
    auto Ca = ops::matmul_avx2 (A, B);
    auto Ct = ops::matmul_threaded(A, B);
    require_close(Cn, Ca, 1e-3);
    require_close(Cn, Ct, 1e-3);
}

TEST_CASE(matmul_identity_exact) {
    std::vector<float> eye(9, 0.0f);
    eye[0] = eye[4] = eye[8] = 1.0f;
    Tensor I({3, 3}, DType::Float32, eye.data(), eye.size() * sizeof(float));
    std::vector<float> av{1, 2, 3, 4, 5, 6};
    Tensor A({2, 3}, DType::Float32, av.data(), av.size() * sizeof(float));
    auto r = ops::matmul_avx2(A, I);
    REQUIRE_NEAR(r.at({0, 0}), 1.0f, 1e-6);
    REQUIRE_NEAR(r.at({0, 1}), 2.0f, 1e-6);
    REQUIRE_NEAR(r.at({1, 2}), 6.0f, 1e-6);
}

TEST_CASE(auto_pick_records_last) {
    auto A = make_random(64, 64, 31);
    auto B = make_random(64, 64, 32);
    (void)ops::matmul(A, B);  // variant=Auto
    auto picked = ops::last_picked_variant();
    REQUIRE(picked != ops::MatmulVariant::Auto);
}