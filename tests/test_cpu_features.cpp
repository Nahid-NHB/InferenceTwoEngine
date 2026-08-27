// tests/test_cpu_features.cpp
// -----------------------------------------------------------------------------
// Tests for the Phase 15 CPU feature probe.
//
// The tests don't try to *force* a particular CPU capability — on every
// machine they run on, they verify the invariants:
//   1. cpu_info() returns a coherent snapshot (every `have_*` matches
//      the field it reads from).
//   2. have_avx512() is true iff all three of F+VL+BW are set.
//   3. hardware_threads() >= 1.
//   4. cpu_feature_summary() is non-empty.
//
// Plus a dispatch-parity test: when AVX2 is built *and* the runtime CPU
// has AVX2, calling the auto-dispatched matmul produces output that
// agrees numerically with the blocked fallback. (The kernels use
// different reduction trees, so we use the same tolerance as the existing
// matmul tests.)
// -----------------------------------------------------------------------------
#include "tinyllm/cpu_features.hpp"

#include "tinyllm/matmul.hpp"
#include "tinyllm/tensor.hpp"

#include "test_helpers.hpp"

#include <cstring>
#include <random>
#include <vector>

TEST_CASE(cpu_info_consistency) {
    const tinyllm::CpuInfo& c = tinyllm::cpu_info();
    REQUIRE(tinyllm::have_sse2()   == c.sse2);
    REQUIRE(tinyllm::have_sse4_2() == c.sse4_2);
    REQUIRE(tinyllm::have_avx()    == c.avx);
    REQUIRE(tinyllm::have_avx2()   == c.avx2);
    REQUIRE(tinyllm::have_fma()    == c.fma);
    REQUIRE(tinyllm::have_bmi2()   == c.bmi2);
    REQUIRE(tinyllm::have_popcnt() == c.popcnt);
    REQUIRE(tinyllm::have_avx512f()  == c.avx512f);
    REQUIRE(tinyllm::have_avx512vl() == c.avx512vl);
    REQUIRE(tinyllm::have_avx512bw() == c.avx512bw);
}

TEST_CASE(avx512_requires_three_features) {
    // have_avx512() iff avx512f && avx512vl && avx512bw.
    const tinyllm::CpuInfo& c = tinyllm::cpu_info();
    REQUIRE(tinyllm::have_avx512() ==
           (c.avx512f && c.avx512vl && c.avx512bw));
}

TEST_CASE(hardware_threads_at_least_one) {
    REQUIRE(tinyllm::hardware_threads() >= 1);
}

TEST_CASE(cpu_feature_summary_nonempty_and_null_terminated) {
    const char* s = tinyllm::cpu_feature_summary();
    REQUIRE(s != nullptr);
    REQUIRE(std::strlen(s) > 0);
    REQUIRE(s[std::strlen(s)] == '\0');
}

#if TINYLLM_ENABLE_AVX2
// The dispatch path that matmul picks when have_avx2() && have_fma()
// matches the explicit MatmulVariant::Avx2 entry point *and* agrees
// numerically with the blocked fallback.
TEST_CASE(matmul_auto_matches_avx2_when_supported) {
    if (!tinyllm::have_avx2() || !tinyllm::have_fma()) {
        // Skip silently on non-AVX2 hardware. We still want the build
        // to link the test.
        return;
    }

    using namespace tinyllm;
    constexpr int64_t M = 64, K = 128, N = 64;
    Tensor a({M, K}, DType::Float32);
    Tensor b({K, N}, DType::Float32);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    for (int64_t i = 0; i < M * K; ++i) a.data_float()[i] = d(rng);
    for (int64_t i = 0; i < K * N; ++i) b.data_float()[i] = d(rng);

    Tensor y_auto    = ops::matmul(a, b, ops::MatmulVariant::Auto);
    Tensor y_avx2    = ops::matmul(a, b, ops::MatmulVariant::Avx2);
    Tensor y_blocked = ops::matmul(a, b, ops::MatmulVariant::Blocked);

    const std::vector<int64_t> expected{M, N};
    REQUIRE(y_auto.shape()    == expected);
    REQUIRE(y_avx2.shape()    == expected);
    REQUIRE(y_blocked.shape() == expected);

    // Auto should select the multi-threaded path on a multi-core host
    // (we have >= 2 cores on every test runner we care about), so the
    // `last_picked_variant()` is Threaded. We don't assert that here —
    // the test would be flaky on a hyper-threaded 1-core sandbox.
    // Just compare numerics.
    for (int64_t i = 0; i < M * N; ++i) {
        REQUIRE_NEAR(y_auto.data_float()[i], y_avx2.data_float()[i], 1e-3f);
        REQUIRE_NEAR(y_auto.data_float()[i], y_blocked.data_float()[i], 1e-3f);
    }
}
#endif  // TINYLLM_ENABLE_AVX2
