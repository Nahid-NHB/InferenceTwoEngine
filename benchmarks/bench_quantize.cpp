// benchmarks/bench_quantize.cpp
// -----------------------------------------------------------------------------
// Benchmark Q4_0 / Q8_0 quantized matvec vs equivalent F32 matmul.
//
// For each shape (M x K) we:
//   - generate random weights (M*K floats) and a random input vector (K)
//   - pack the weights to Q4_0 / Q8_0
//   - time: F32 matmul (one row at a time, single-precision reference),
//           Q4_0×F32 matvec (AVX-512 fused if available, else AVX2
//           fused, else reference),
//           Q8_0×F32 matvec (currently the dequant-then-FMA reference).
//
// The Q4_0 numbers demonstrate the benefit of fusing dequant with the dot
// product (no temp row buffer). Q8_0 currently falls back to the reference;
// that's the next thing to optimize if Q8_0 matters.
// -----------------------------------------------------------------------------
#include "tinyllm/matmul.hpp"
#include "tinyllm/quantize.hpp"
#include "tinyllm/tensor.hpp"

#include <chrono>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

using namespace tinyllm;
using clk = std::chrono::high_resolution_clock;

int main() {
    std::printf("# build: AVX2=%d AVX-512=%d threads=%d\n",
                ops::have_avx2() ? 1 : 0,
                ops::have_avx512() ? 1 : 0,
                ops::hardware_threads());
    struct Shape { int64_t M, K; };
    std::vector<Shape> shapes = {
        {128, 256},
        {256, 512},
        {512, 1024},
        {1024, 2048},
    };

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::printf("format,M,K,f32_ms,qmat_ms,speedup\n");
    for (const auto& s : shapes) {
        int64_t M = s.M;
        int64_t K = s.K;
        std::size_t mk = static_cast<std::size_t>(M * K);
        std::vector<float> w(mk), x(static_cast<std::size_t>(K));
        for (auto& v : w) v = dist(rng);
        for (auto& v : x) v = dist(rng);

        // F32 reference: a (M,K) × x (K,1) → (M,1).
        Tensor W({M, K}, DType::Float32, w.data(), mk * sizeof(float));
        Tensor X({K, 1}, DType::Float32, x.data(), static_cast<std::size_t>(K) * sizeof(float));

        // Warmup.
        auto warm = ops::matmul(W, X, ops::MatmulVariant::Avx2);
        (void)warm;

        auto tf0 = clk::now();
        auto yf = ops::matmul(W, X, ops::MatmulVariant::Avx2);
        auto tf1 = clk::now();
        (void)yf;
        double f32_ms = std::chrono::duration<double, std::milli>(tf1 - tf0).count();

        // ---- Q4_0
        auto packed_q4 = quantize_q4_0(w.data(), M * K);
        std::vector<float> y_q4(static_cast<std::size_t>(M));
        // Warmup
        matmul_q4_0_f32(packed_q4.data(), M, K, x.data(), y_q4.data());
        auto tq0 = clk::now();
        matmul_q4_0_f32(packed_q4.data(), M, K, x.data(), y_q4.data());
        auto tq1 = clk::now();
        double q4_ms = std::chrono::duration<double, std::milli>(tq1 - tq0).count();
        std::printf("Q4_0,%ld,%ld,%.3f,%.3f,%.2fx\n",
                    M, K, f32_ms, q4_ms, f32_ms / q4_ms);

        // ---- Q4_0 AVX-512 vs AVX2 head-to-head.
        // When both ISAs are compiled in, the static dispatch picks the
        // best one (AVX-512). Call the AVX2 kernel directly here to put
        // a number on the AVX-512 speedup over AVX2.
#if TINYLLM_ENABLE_AVX2
        std::vector<float> y_q4_avx2(static_cast<std::size_t>(M));
        // Warmup
        ops::matvec_q4_0_f32_avx2(packed_q4.data(), M, K, x.data(), y_q4_avx2.data());
        auto ta0 = clk::now();
        ops::matvec_q4_0_f32_avx2(packed_q4.data(), M, K, x.data(), y_q4_avx2.data());
        auto ta1 = clk::now();
        double q4_avx2_ms = std::chrono::duration<double, std::milli>(ta1 - ta0).count();
        std::printf("Q4_0_avx2,%ld,%ld,,%.3f,\n", M, K, q4_avx2_ms);
#endif
#if TINYLLM_ENABLE_AVX512
        std::vector<float> y_q4_avx512(static_cast<std::size_t>(M));
        // Warmup
        ops::matvec_q4_0_f32_avx512(packed_q4.data(), M, K, x.data(), y_q4_avx512.data());
        auto t512_0 = clk::now();
        ops::matvec_q4_0_f32_avx512(packed_q4.data(), M, K, x.data(), y_q4_avx512.data());
        auto t512_1 = clk::now();
        double q4_avx512_ms = std::chrono::duration<double, std::milli>(t512_1 - t512_0).count();
        std::printf("Q4_0_avx512,%ld,%ld,,%.3f,\n", M, K, q4_avx512_ms);
        std::printf("# speedup avx512 vs avx2: %.2fx at (%ld,%ld)\n",
                    q4_avx2_ms / q4_avx512_ms, M, K);
#endif

        // ---- Q8_0
        auto packed_q8 = quantize_q8_0(w.data(), M * K);
        std::vector<float> y_q8(static_cast<std::size_t>(M));
        matmul_q8_0_f32(packed_q8.data(), M, K, x.data(), y_q8.data());
        auto t8_0 = clk::now();
        matmul_q8_0_f32(packed_q8.data(), M, K, x.data(), y_q8.data());
        auto t8_1 = clk::now();
        double q8_ms = std::chrono::duration<double, std::milli>(t8_1 - t8_0).count();
        std::printf("Q8_0,%ld,%ld,%.3f,%.3f,%.2fx\n",
                    M, K, f32_ms, q8_ms, f32_ms / q8_ms);
    }
    return 0;
}
