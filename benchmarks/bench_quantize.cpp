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
//           Q4_K×F32 matvec (fused AVX2 if compiled in, else reference),
//           Q6_K×F32 matvec (fused AVX2 if compiled in, else reference),
//           Q8_0×F32 matvec (currently the dequant-then-FMA reference).
//
// The Q4_0 / Q4_K / Q6_K numbers demonstrate the benefit of fusing
// dequant with the dot product (no temp row buffer). Q8_0 currently
// falls back to the reference; that's the next thing to optimize if
// Q8_0 matters.
// -----------------------------------------------------------------------------
#include "tinyllm/matmul.hpp"
#include "tinyllm/quantize.hpp"
#include "tinyllm/tensor.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
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

        // ---- Q4_K (Phase 14 fused AVX2 kernel). We construct a random
        // but well-formed Q4_K buffer per row: every super-block has
        // d=0.05, dmin=0, scales=[15..] packed into 12 bytes, qs filled
        // with nibbles encoding (w / (d * sc)) rounded. This is not
        // llama.cpp bit-exact quant but it's a valid K-quant layout, so
        // the fused dequant+FMA kernel exercises its full data path.
        const int64_t sb_per_row = K / 256;  // kQK_K = 256
        if (sb_per_row * 256 == K) {
            constexpr std::size_t kQ4_KBlockBytes = 144;
            std::size_t bytes_per_row_k = static_cast<std::size_t>(sb_per_row) *
                                           kQ4_KBlockBytes;
            std::vector<uint8_t> qmat_k(static_cast<std::size_t>(M) *
                                         bytes_per_row_k);
            const float d_global_k = 0.05f;
            for (int64_t m = 0; m < M; ++m) {
                for (int64_t sb = 0; sb < sb_per_row; ++sb) {
                    uint8_t* p = qmat_k.data() + m * bytes_per_row_k +
                                 sb * kQ4_KBlockBytes;
                    uint16_t d_h  = quantize_f32_to_f16(d_global_k);
                    uint16_t dm_h = quantize_f32_to_f16(0.0f);
                    std::memcpy(p,     &d_h,  sizeof(uint16_t));
                    std::memcpy(p + 2, &dm_h, sizeof(uint16_t));
                    // Simple scale layout: all 8 sub-blocks get sc=15, m=0.
                    // Encoding per llama.cpp's get_scale_min_k4: 12 bytes,
                    // 4-bit packed, sc[n] in low nibble, m[n] in high nibble.
                    // For sc=15, m=0, the packed byte is just 15 | (0<<4) = 15.
                    for (int i = 0; i < 12; ++i) p[4 + i] = 15;
                    uint8_t* qs = p + 4 + 12;
                    for (int chunk = 0; chunk < 4; ++chunk) {
                        for (int j = 0; j < 32; ++j) {
                            int64_t idx_lo = chunk * 64 + j;
                            int64_t idx_hi = chunk * 64 + j + 32;
                            int lo = std::clamp(static_cast<int>(
                                std::round(w[m * K + sb * 256 + idx_lo] /
                                           (d_global_k * 15.0f))), 0, 15);
                            int hi = std::clamp(static_cast<int>(
                                std::round(w[m * K + sb * 256 + idx_hi] /
                                           (d_global_k * 15.0f))), 0, 15);
                            qs[chunk * 32 + j] =
                                static_cast<uint8_t>(lo | (hi << 4));
                        }
                    }
                }
            }
            std::vector<float> y_q4k(static_cast<std::size_t>(M));
            matmul_q4_K_f32(qmat_k.data(), M, K, x.data(), y_q4k.data());
            auto tk0 = clk::now();
            matmul_q4_K_f32(qmat_k.data(), M, K, x.data(), y_q4k.data());
            auto tk1 = clk::now();
            double q4k_ms = std::chrono::duration<double, std::milli>(tk1 - tk0).count();
            std::printf("Q4_K,%ld,%ld,%.3f,%.3f,%.2fx\n",
                        M, K, f32_ms, q4k_ms, f32_ms / q4k_ms);
        }

        // ---- Q6_K (Phase 14 fused AVX2 kernel). Random valid byte
        // buffer (super-block is 210 bytes: 2 d, 16 scales int8,
        // 128 ql, 64 qh). The kernel doesn't care about accuracy, just
        // a valid byte layout, so random bytes are fine.
        if (sb_per_row * 256 == K) {
            constexpr std::size_t kQ6_KBlockBytes = 210;
            std::size_t bytes_per_row_6 = static_cast<std::size_t>(sb_per_row) *
                                           kQ6_KBlockBytes;
            std::vector<uint8_t> qmat_6(static_cast<std::size_t>(M) *
                                         bytes_per_row_6);
            // Use a non-uniform byte distribution so the kernel sees
            // realistic values: d in [0.01, 0.1] (small positives),
            // scales int8 in [-64, 64] (centered), qs/qh uniform 0..255.
            std::mt19937 krng(12345);
            std::uniform_real_distribution<float> d_dist(0.01f, 0.1f);
            std::uniform_int_distribution<int> sb_dist(-64, 64);
            std::uniform_int_distribution<int> byte_dist(0, 255);
            for (int64_t m = 0; m < M; ++m) {
                for (int64_t sb = 0; sb < sb_per_row; ++sb) {
                    uint8_t* p = qmat_6.data() + m * bytes_per_row_6 +
                                 sb * kQ6_KBlockBytes;
                    uint16_t d_h = quantize_f32_to_f16(d_dist(krng));
                    std::memcpy(p, &d_h, sizeof(uint16_t));
                    for (int i = 0; i < 16; ++i)
                        p[2 + i] = static_cast<uint8_t>(sb_dist(krng) & 0xFF);
                    for (int i = 0; i < 128; ++i)
                        p[2 + 16 + i] = static_cast<uint8_t>(byte_dist(krng));
                    for (int i = 0; i < 64; ++i)
                        p[2 + 16 + 128 + i] = static_cast<uint8_t>(byte_dist(krng));
                }
            }
            std::vector<float> y_q6k(static_cast<std::size_t>(M));
            matmul_q6_K_f32(qmat_6.data(), M, K, x.data(), y_q6k.data());
            auto t6k0 = clk::now();
            matmul_q6_K_f32(qmat_6.data(), M, K, x.data(), y_q6k.data());
            auto t6k1 = clk::now();
            double q6k_ms = std::chrono::duration<double, std::milli>(t6k1 - t6k0).count();
            std::printf("Q6_K,%ld,%ld,%.3f,%.3f,%.2fx\n",
                        M, K, f32_ms, q6k_ms, f32_ms / q6k_ms);
        }
    }
    return 0;
}
