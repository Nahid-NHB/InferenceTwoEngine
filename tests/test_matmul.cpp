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
#include "tinyllm/cpu_features.hpp"
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

// -----------------------------------------------------------------------------
// Phase 16 — AVX-512 F32 matmul
//
// The kernel is a direct port of mm_avx2_kernel to __m512. If the build
// emitted AVX-512 we run the kernel and confirm it agrees with the
// AVX2 / naive references; if not, we silently skip (the Phase 15
// dispatch already verifies the runtime fallback path).
// -----------------------------------------------------------------------------
#if TINYLLM_ENABLE_AVX512

TEST_CASE(avx512_matmul_agrees_on_random) {
    auto A = make_random(96, 128, 41);
    auto B = make_random(128, 96, 42);
    auto Cn  = ops::matmul_naive  (A, B);
    auto Ca  = ops::matmul_avx2   (A, B);
    auto C512 = ops::matmul_avx512(A, B);
    require_close(Cn, Ca,   1e-3);
    require_close(Cn, C512, 1e-3);
}

TEST_CASE(avx512_matmul_agrees_square_large) {
    // Sizes chosen to clear the cache-blocked L1 tile (MB=96, NB=256,
    // KB=64) multiple times so we exercise outer-loop reuse, not just
    // the inner micro-kernel.
    auto A = make_random(384, 384, 51);
    auto B = make_random(384, 384, 52);
    auto Cn   = ops::matmul_naive  (A, B);
    auto Ca   = ops::matmul_avx2   (A, B);
    auto C512 = ops::matmul_avx512(A, B);
    require_close(Cn, Ca,   1e-2f);
    require_close(Cn, C512, 1e-2f);
}

TEST_CASE(avx512_matmul_agrees_non_square) {
    // Realistic transformer-ish shapes: small M (decode-step eq),
    // wide K (hidden=4096), wide N (vocab).
    constexpr int64_t M = 8;
    constexpr int64_t K = 256;
    constexpr int64_t N = 256;
    auto A = make_random(M, K, 61);
    auto B = make_random(K, N, 62);
    auto Cn   = ops::matmul_naive  (A, B);
    auto Ca   = ops::matmul_avx2   (A, B);
    auto C512 = ops::matmul_avx512(A, B);
    require_close(Cn, Ca,   1e-3);
    require_close(Cn, C512, 1e-3);
}

TEST_CASE(avx512_matmul_agrees_tail_sizes) {
    // Sizes not divisible by the AVX-512 micro-tile (6 rows, 32 cols).
    // The fallback scalar tail inside mm_avx512_kernel must still
    // produce correct results.
    auto A = make_random(70, 130, 71);
    auto B = make_random(130, 50, 72);
    auto Cn   = ops::matmul_naive(A, B);
    auto Ca   = ops::matmul_avx2 (A, B);
    auto C512 = ops::matmul_avx512(A, B);
    require_close(Cn, Ca,   1e-3);
    require_close(Cn, C512, 1e-3);
}

TEST_CASE(avx512_matmul_identity_exact) {
    std::vector<float> eye(64, 0.0f);
    for (int64_t i = 0; i < 8; ++i) eye[static_cast<std::size_t>(i * 8 + i)] = 1.0f;
    Tensor I({8, 8}, DType::Float32, eye.data(), eye.size() * sizeof(float));
    auto A = make_random(8, 8, 81);
    auto R = ops::matmul_avx512(A, I);
    for (int64_t i = 0; i < 8; ++i) {
        for (int64_t j = 0; j < 8; ++j) {
            float want = (i == j) ? A.at({i, j}) : A.at({i, j});
            (void)want;
        }
    }
    // The point: A * I == A exactly because A * I is just A copied.
    // We compare row-by-row to the original A.
    for (int64_t i = 0; i < 8; ++i) {
        for (int64_t j = 0; j < 8; ++j) {
            REQUIRE_NEAR(R.at({i, j}), A.at({i, j}), 1e-6f);
        }
    }
}

TEST_CASE(avx512_pick_prefers_avx512_when_available) {
    // We can't make the runtime CPU expose features it lacks, but we
    // can exercise `matmul(..., Auto)` and confirm the recorded pick
    // is consistent with the compile-time × runtime gates.
    auto A = make_random(96, 96, 91);
    auto B = make_random(96, 96, 92);
    (void)ops::matmul(A, B, ops::MatmulVariant::Auto);
    const auto picked = ops::last_picked_variant();
    // When AVX-512 is built AND the host has it, Auto picks AVX-512
    // (no multi-threading check here since it depends on cores); when
    // only AVX2 is present it picks AVX2; otherwise Blocked.
    if (tinyllm::have_avx512()) {
        REQUIRE(picked == ops::MatmulVariant::Avx512 ||
                picked == ops::MatmulVariant::Threaded);
    } else if (tinyllm::have_avx2()) {
        REQUIRE(picked == ops::MatmulVariant::Avx2 ||
                picked == ops::MatmulVariant::Threaded);
    } else {
        REQUIRE(picked == ops::MatmulVariant::Blocked ||
                picked == ops::MatmulVariant::Threaded);
    }
}

#endif  // TINYLLM_ENABLE_AVX512