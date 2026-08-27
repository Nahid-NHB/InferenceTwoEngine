// src/quantize.cpp
// -----------------------------------------------------------------------------
// Block-wise quantization (Q8_0, Q4_0) + reference quantized matmul.
//
// See include/tinyllm/quantize.hpp for the spec.
//
// Round-trip error budget (informal):
//   - Q8_0: relative error ≈ 1/127 ≈ 0.8% per value.
//   - Q4_0: relative error ≈ 1/15 ≈ 6.7% per value (after the [-8, 7]
//     re-centering).
//
// We do *not* yet implement a fused dequant-dot kernel; that lands in
// Phase 9. The matmul_q*_f32 functions in this file dequantize each row
// and call into the existing F32 matmul, so they're a baseline that
// future fused kernels can replace.
// -----------------------------------------------------------------------------
#include "tinyllm/quantize.hpp"

#include "tinyllm/cpu_features.hpp"
#include "tinyllm/matmul.hpp"
#include "tinyllm/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace tinyllm {

// Forward declaration for the AVX2 / AVX-512 fused Q4_0 paths (see
// matmul.cpp). Defined in tinyllm::ops::matvec_q4_0_f32_avx{2,512}; we
// call them from matmul_q4_0_f32 when the matching ISA is enabled.
// Phase 14: same idea for Q4_K / Q6_K.
namespace ops {
void matvec_q4_0_f32_avx2(const uint8_t* qmat, int64_t M, int64_t K,
                          const float* x, float* y);
void matvec_q4_0_f32_avx512(const uint8_t* qmat, int64_t M, int64_t K,
                            const float* x, float* y);
void matvec_q4_K_f32_avx2(const uint8_t* qmat, int64_t M, int64_t K,
                          const float* x, float* y);
void matvec_q6_K_f32_avx2(const uint8_t* qmat, int64_t M, int64_t K,
                          const float* x, float* y);
}

// =============================================================================
// Half <-> single conversion (independent of gguf.cpp's F16 path).
// =============================================================================
float quantize_f16_to_f32(uint16_t h) noexcept {
    uint32_t sign     = (h >> 15) & 0x1;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;
    uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign << 31;
        } else {
            int e = -14;
            while ((mantissa & 0x400) == 0) { mantissa <<= 1; ++e; }
            mantissa &= 0x3FF;
            bits = (sign << 31) | ((e + 127) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        bits = (sign << 31) | (0xFF << 23) | (mantissa << 13);
    } else {
        bits = (sign << 31) | ((exponent - 15 + 127) << 23) | (mantissa << 13);
    }
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

uint16_t quantize_f32_to_f16(float x) noexcept {
    uint32_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    uint32_t sign     = (bits >> 31) & 0x1;
    int32_t  exponent = static_cast<int32_t>((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = bits & 0x7FFFFF;

    if (exponent <= 0) {
        if (exponent < -10) {
            // Too small to represent; round to zero (we don't bother with
            // subnormals here; quantization precision is the bottleneck
            // anyway).
            return static_cast<uint16_t>(sign << 15);
        }
        // Subnormal: shift mantissa, round.
        mantissa = (mantissa | 0x800000) >> static_cast<uint32_t>(1 - exponent);
        // Round-to-nearest-even on the bottom bit we're dropping.
        uint32_t round_bit = 1u << static_cast<uint32_t>(13 - exponent);
        uint32_t sticky    = mantissa & (round_bit - 1);
        mantissa += (sticky > round_bit / 2 ||
                     (sticky == round_bit / 2 && (mantissa & round_bit)))
                    ? round_bit
                    : 0u;
        mantissa &= ~(round_bit - 1);
        if (mantissa & 0x800) {
            // Rounded up to a normal.
            mantissa = 0;
            exponent = 1;
        }
        return static_cast<uint16_t>((sign << 15) |
                                     (static_cast<uint32_t>(exponent) << 10) |
                                     (mantissa & 0x3FF));
    }
    if (exponent >= 31) {
        // Inf / NaN — saturate to +/- inf.
        return static_cast<uint16_t>((sign << 15) | (0x1F << 10));
    }

    // Normal case: drop the bottom 13 mantissa bits with round-to-nearest.
    uint32_t round_bit = 1u << 13;
    uint32_t sticky    = mantissa & (round_bit - 1);
    mantissa += (sticky > round_bit / 2 ||
                 (sticky == round_bit / 2 && (mantissa & round_bit)))
                ? round_bit
                : 0u;
    mantissa >>= 13;
    if (mantissa & 0x400) {
        mantissa = 0;
        exponent += 1;
        if (exponent >= 31) {
            return static_cast<uint16_t>((sign << 15) | (0x1F << 10));
        }
    }
    return static_cast<uint16_t>((sign << 15) |
                                 (static_cast<uint32_t>(exponent) << 10) |
                                 (mantissa & 0x3FF));
}

// =============================================================================
// Q8_0
// =============================================================================
Q8_0Block quantize_q8_0_block(const float* src) noexcept {
    // Find absmax.
    float amax = 0.0f;
    for (int64_t i = 0; i < kQ8_0BlockSize; ++i) {
        float a = std::fabs(src[i]);
        if (a > amax) amax = a;
    }
    Q8_0Block b;
    if (amax == 0.0f) {
        b.scale = 0.0f;
        b.qs.fill(0);
        return b;
    }
    float scale = amax / 127.0f;
    float inv   = 1.0f / scale;
    b.scale = scale;
    for (int64_t i = 0; i < kQ8_0BlockSize; ++i) {
        float q = src[i] * inv;
        // Round-half-away-from-zero, then clamp.
        int32_t qi = static_cast<int32_t>(q + (q >= 0.0f ? 0.5f : -0.5f));
        if (qi >  127) qi =  127;
        if (qi < -127) qi = -127;
        b.qs[static_cast<std::size_t>(i)] = static_cast<int8_t>(qi);
    }
    return b;
}

void dequantize_q8_0_block(const Q8_0Block& b, float* dst) noexcept {
    for (int64_t i = 0; i < kQ8_0BlockSize; ++i) {
        dst[i] = static_cast<float>(b.qs[static_cast<std::size_t>(i)]) * b.scale;
    }
}

std::vector<uint8_t> quantize_q8_0(const float* src, int64_t n) {
    if (n % kQ8_0BlockSize != 0) {
        throw std::runtime_error("quantize_q8_0: n must be divisible by 32");
    }
    int64_t nblocks = n / kQ8_0BlockSize;
    std::vector<uint8_t> out(static_cast<std::size_t>(nblocks * kQ8_0BlockBytes));
    for (int64_t bi = 0; bi < nblocks; ++bi) {
        Q8_0Block b = quantize_q8_0_block(src + bi * kQ8_0BlockSize);
        uint8_t* p = out.data() + bi * kQ8_0BlockBytes;
        uint16_t sh = quantize_f32_to_f16(b.scale);
        std::memcpy(p, &sh, sizeof(sh));
        std::memcpy(p + 2, b.qs.data(), kQ8_0BlockSize);
    }
    return out;
}

void dequantize_q8_0(const uint8_t* packed, int64_t n, float* dst) {
    if (n % kQ8_0BlockSize != 0) {
        throw std::runtime_error("dequantize_q8_0: n must be divisible by 32");
    }
    int64_t nblocks = n / kQ8_0BlockSize;
    for (int64_t bi = 0; bi < nblocks; ++bi) {
        const uint8_t* p = packed + bi * kQ8_0BlockBytes;
        uint16_t sh;
        std::memcpy(&sh, p, sizeof(sh));
        Q8_0Block b;
        b.scale = quantize_f16_to_f32(sh);
        std::memcpy(b.qs.data(), p + 2, kQ8_0BlockSize);
        dequantize_q8_0_block(b, dst + bi * kQ8_0BlockSize);
    }
}

// =============================================================================
// Q4_0
// =============================================================================
Q4_0Block quantize_q4_0_block(const float* src) noexcept {
    float amax = 0.0f;
    for (int64_t i = 0; i < kQ4_0BlockSize; ++i) {
        float a = std::fabs(src[i]);
        if (a > amax) amax = a;
    }
    Q4_0Block b;
    b.qs.fill(0);
    if (amax == 0.0f) {
        b.scale = 0.0f;
        return b;
    }
    // Map [-amax, +amax] to [-8, +7]. The half-open range means we use
    // 15 levels (since we re-center). Scale = amax / 7.
    float scale = amax / 7.0f;
    float inv   = 1.0f / scale;
    b.scale = scale;
    for (int64_t i = 0; i < kQ4_0BlockSize; ++i) {
        float q = src[i] * inv;
        int32_t qi = static_cast<int32_t>(q + (q >= 0.0f ? 0.5f : -0.5f));
        if (qi >  7) qi =  7;
        if (qi < -8) qi = -8;
        uint8_t nibble = static_cast<uint8_t>(qi + 8);  // re-center to [0, 15]
        if ((i & 1) == 0) {
            b.qs[static_cast<std::size_t>(i / 2)] = static_cast<uint8_t>(nibble & 0x0F);
        } else {
            b.qs[static_cast<std::size_t>(i / 2)] |=
                static_cast<uint8_t>((nibble & 0x0F) << 4);
        }
    }
    return b;
}

void dequantize_q4_0_block(const Q4_0Block& b, float* dst) noexcept {
    for (int64_t i = 0; i < kQ4_0BlockSize; ++i) {
        uint8_t nibble = ((i & 1) == 0)
                       ? (b.qs[static_cast<std::size_t>(i / 2)] & 0x0F)
                       : ((b.qs[static_cast<std::size_t>(i / 2)] >> 4) & 0x0F);
        int32_t qi = static_cast<int32_t>(nibble) - 8;
        dst[i] = static_cast<float>(qi) * b.scale;
    }
}

std::vector<uint8_t> quantize_q4_0(const float* src, int64_t n) {
    if (n % kQ4_0BlockSize != 0) {
        throw std::runtime_error("quantize_q4_0: n must be divisible by 32");
    }
    int64_t nblocks = n / kQ4_0BlockSize;
    std::vector<uint8_t> out(static_cast<std::size_t>(nblocks * kQ4_0BlockBytes));
    for (int64_t bi = 0; bi < nblocks; ++bi) {
        Q4_0Block b = quantize_q4_0_block(src + bi * kQ4_0BlockSize);
        uint8_t* p = out.data() + bi * kQ4_0BlockBytes;
        uint16_t sh = quantize_f32_to_f16(b.scale);
        std::memcpy(p, &sh, sizeof(sh));
        std::memcpy(p + 2, b.qs.data(), 16);
    }
    return out;
}

void dequantize_q4_0(const uint8_t* packed, int64_t n, float* dst) {
    if (n % kQ4_0BlockSize != 0) {
        throw std::runtime_error("dequantize_q4_0: n must be divisible by 32");
    }
    int64_t nblocks = n / kQ4_0BlockSize;
    for (int64_t bi = 0; bi < nblocks; ++bi) {
        const uint8_t* p = packed + bi * kQ4_0BlockBytes;
        uint16_t sh;
        std::memcpy(&sh, p, sizeof(sh));
        Q4_0Block b;
        b.scale = quantize_f16_to_f32(sh);
        std::memcpy(b.qs.data(), p + 2, 16);
        dequantize_q4_0_block(b, dst + bi * kQ4_0BlockSize);
    }
}

// =============================================================================
// Reference quantized matmul. Phase 9 adds an AVX2-fused Q4_0 path
// (see src/matmul.cpp::matvec_q4_0_f32_avx2). When built with AVX2
// support, we use the fused kernel here. Otherwise we fall back to a
// dequant-then-FMA reference.
// =============================================================================
namespace {

void matmul_q8_0_f32_reference(const uint8_t* qmat, int64_t M, int64_t K,
                                const float* x, float* y) {
    if (K % kQ8_0BlockSize != 0) {
        throw std::runtime_error("matmul_q8_0_f32: K must be divisible by 32");
    }
    std::vector<float> row(static_cast<std::size_t>(K));
    std::size_t bytes_per_row = static_cast<std::size_t>(K / kQ8_0BlockSize) *
                                kQ8_0BlockBytes;
    for (int64_t m = 0; m < M; ++m) {
        dequantize_q8_0(qmat + m * bytes_per_row, K, row.data());
        double acc = 0.0;
        for (int64_t k = 0; k < K; ++k) {
            acc += static_cast<double>(row[static_cast<std::size_t>(k)]) *
                   static_cast<double>(x[k]);
        }
        y[m] = static_cast<float>(acc);
    }
}

// Phase 15: always available as the runtime fallback (no longer guarded
// by the SIMD compile-time flags — the dispatch in matmul_q4_0_f32 now
// chooses between this and the AVX kernels at runtime).
void matmul_q4_0_f32_reference(const uint8_t* qmat, int64_t M, int64_t K,
                                const float* x, float* y) {
    if (K % kQ4_0BlockSize != 0) {
        throw std::runtime_error("matmul_q4_0_f32: K must be divisible by 32");
    }
    std::vector<float> row(static_cast<std::size_t>(K));
    std::size_t bytes_per_row = static_cast<std::size_t>(K / kQ4_0BlockSize) *
                                kQ4_0BlockBytes;
    for (int64_t m = 0; m < M; ++m) {
        dequantize_q4_0(qmat + m * bytes_per_row, K, row.data());
        double acc = 0.0;
        for (int64_t k = 0; k < K; ++k) {
            acc += static_cast<double>(row[static_cast<std::size_t>(k)]) *
                   static_cast<double>(x[k]);
        }
        y[m] = static_cast<float>(acc);
    }
}

// (Forward declaration for the AVX2 fused Q4_0 path lives at the top of
// this file, outside this anonymous namespace.)

}  // namespace

void matmul_q4_0_f32(const uint8_t* qmat, int64_t M, int64_t K,
                     const float* x, float* y) {
    // Phase 15: runtime dispatch.
    //   - If the build emitted AVX-512 (TINYLLM_ENABLE_AVX512=1) AND the
    //     CPU exposes AVX-512F+VL+BW at runtime, use the AVX-512 kernel.
    //   - Else, if the build emitted AVX2 AND the CPU has AVX2+FMA at
    //     runtime, use the AVX2 kernel.
    //   - Else, dequantize-on-the-fly reference.
    // The compile-time flag is still an upper-bound gate — a build with
    // TINYLLM_ENABLE_AVX2=OFF cannot accidentally call the AVX2 kernel.
#if TINYLLM_ENABLE_AVX512
    if (tinyllm::have_avx512()) {
        tinyllm::ops::matvec_q4_0_f32_avx512(qmat, M, K, x, y);
        return;
    }
#endif
#if TINYLLM_ENABLE_AVX2
    if (tinyllm::have_avx2() && tinyllm::have_fma()) {
        tinyllm::ops::matvec_q4_0_f32_avx2(qmat, M, K, x, y);
        return;
    }
#endif
    matmul_q4_0_f32_reference(qmat, M, K, x, y);
}

void matmul_q8_0_f32(const uint8_t* qmat, int64_t M, int64_t K,
                     const float* x, float* y) {
    matmul_q8_0_f32_reference(qmat, M, K, x, y);
}

// =============================================================================
// K-quants (Q4_K, Q5_K, Q6_K)
//
// These formats are what real-world Llama GGUFs use, so dequantizing them
// is the load path to actually running a real model. We port the
// reference dequantize functions from llama.cpp / ggml (Apache-2.0).
// The byte layouts and scale-min packing schemes are not documented in
// the GGUF spec — they live in ggml-common.h — so we follow the
// reference implementation exactly.
//
// Format details:
//
//   QK_K = 256 (super-block size, 8 sub-blocks of 32).
//
//   Q4_K super-block: 2 d (f16) | 2 dmin (f16) | 12 scales |
//                     128 qs (4-bit nibbles, packed)
//      - dequant: y[i] = d * sc * q - dmin * m
//        where (sc, m) are the 6-bit sub-block (scale, min), unpacked
//        from the 12-byte scales array via get_scale_min_k4.
//
//   Q5_K super-block: same as Q4_K + 32 qh (high-bit nibbles, 1 bit per
//                     element).
//      - dequant: y[i] = d * sc * (lo + (hi ? 16 : 0)) - dmin * m
//        where lo = 4-bit value from qs, hi = the corresponding bit
//        from qh. The two bits-of-interest per 64-element chunk are
//        tracked by (u1, u2) bitmasks that shift left every 64 elts.
//
//   Q6_K super-block: 16 sub-blocks of 16 elements; each sub-block
//                     carries its own int8 scale.
//      Layout: 128 ql (lower 4 bits) | 64 qh (upper 2 bits) |
//              16 scales (int8) | 2 d (f16).
//      - dequant: y[i] = d * sc[is] * ((ql_lo|qh_hi2) - 32)
//        where (ql_lo, qh_hi2) gives a 6-bit value centered at 32.
//
// All three paths are pure dequantize; we do not implement quantization
// (K-means search) — that lives in llama.cpp's `quantize_row_q*_K_impl`
// and isn't needed for read-only model loading.
// =============================================================================
namespace {

// Unpack the (sc, m) 6-bit pair for sub-block index `j` from the
// 12-byte scale table of a Q4_K / Q5_K super-block. This is the
// reference `get_scale_min_k4` from llama.cpp verbatim.
inline void get_scale_min_k4(int j, const uint8_t* q,
                             uint8_t* d, uint8_t* m) noexcept {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = static_cast<uint8_t>((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        *m = static_cast<uint8_t>((q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4));
    }
}

}  // namespace

void dequantize_q4_K(const uint8_t* packed, int64_t n, float* dst) {
    if (n % kQK_K != 0) {
        throw std::runtime_error("dequantize_q4_K: n must be divisible by 256");
    }
    const int64_t nb = n / kQK_K;
    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* q = packed + i * kQ4_KBlockBytes;
        uint16_t d_h, dmin_h;
        std::memcpy(&d_h,    q,       sizeof(uint16_t));
        std::memcpy(&dmin_h, q + 2,   sizeof(uint16_t));
        const float d   = quantize_f16_to_f32(d_h);
        const float dm  = quantize_f16_to_f32(dmin_h);
        const uint8_t* sc = q + 4;
        const uint8_t* qs = q + 4 + kKScaleSize;
        const int64_t base = i * kQK_K;

        int is = 0;
        uint8_t sc_l, m_l;
        for (int j = 0; j < kQK_K; j += 64) {
            get_scale_min_k4(is + 0, sc, &sc_l, &m_l);
            const float d1 = d * sc_l;
            const float m1 = dm * m_l;
            get_scale_min_k4(is + 1, sc, &sc_l, &m_l);
            const float d2 = d * sc_l;
            const float m2 = dm * m_l;
            for (int l = 0; l < 32; ++l) {
                dst[base + j + l]      = d1 * static_cast<float>(qs[l] & 0xF) - m1;
                dst[base + j + l + 32] = d2 * static_cast<float>(qs[l] >> 4) - m2;
            }
            qs += 32;
            is += 2;
        }
    }
}

void dequantize_q5_K(const uint8_t* packed, int64_t n, float* dst) {
    if (n % kQK_K != 0) {
        throw std::runtime_error("dequantize_q5_K: n must be divisible by 256");
    }
    const int64_t nb = n / kQK_K;
    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* q = packed + i * kQ5_KBlockBytes;
        uint16_t d_h, dmin_h;
        std::memcpy(&d_h,    q,       sizeof(uint16_t));
        std::memcpy(&dmin_h, q + 2,   sizeof(uint16_t));
        const float d   = quantize_f16_to_f32(d_h);
        const float dm  = quantize_f16_to_f32(dmin_h);
        const uint8_t* sc = q + 4;
        const uint8_t* ql = q + 4 + kKScaleSize;
        const uint8_t* qh = ql + kQK_K / 2;
        const int64_t base = i * kQK_K;

        int is = 0;
        uint8_t sc_l, m_l;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < kQK_K; j += 64) {
            get_scale_min_k4(is + 0, sc, &sc_l, &m_l);
            const float d1 = d * sc_l;
            const float m1 = dm * m_l;
            get_scale_min_k4(is + 1, sc, &sc_l, &m_l);
            const float d2 = d * sc_l;
            const float m2 = dm * m_l;
            for (int l = 0; l < 32; ++l) {
                const int lo_l = ql[l] & 0xF;
                const int hi_l = (qh[l] & u1) ? 16 : 0;
                dst[base + j + l]      = d1 * static_cast<float>(lo_l + hi_l) - m1;

                const int lo_h = ql[l] >> 4;
                const int hi_h = (qh[l] & u2) ? 16 : 0;
                dst[base + j + l + 32] = d2 * static_cast<float>(lo_h + hi_h) - m2;
            }
            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

void dequantize_q6_K(const uint8_t* packed, int64_t n, float* dst) {
    if (n % kQK_K != 0) {
        throw std::runtime_error("dequantize_q6_K: n must be divisible by 256");
    }
    const int64_t nb = n / kQK_K;
    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* q = packed + i * kQ6_KBlockBytes;
        const uint8_t*  ql = q;
        const uint8_t*  qh = ql + kQK_K / 2;
        const int8_t*   sc = reinterpret_cast<const int8_t*>(qh + kQK_K / 4);
        uint16_t d_h;
        std::memcpy(&d_h, sc + kQK_K / 16, sizeof(uint16_t));
        const float d = quantize_f16_to_f32(d_h);
        const int64_t base = i * kQK_K;

        for (int n_off = 0; n_off < kQK_K; n_off += 128) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int8_t q1 = static_cast<int8_t>(
                    ((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32);
                const int8_t q2 = static_cast<int8_t>(
                    ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32);
                const int8_t q3 = static_cast<int8_t>(
                    ((ql[l +  0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32);
                const int8_t q4 = static_cast<int8_t>(
                    ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
                dst[base + n_off + l +  0] = d * sc[is + 0] * static_cast<float>(q1);
                dst[base + n_off + l + 32] = d * sc[is + 2] * static_cast<float>(q2);
                dst[base + n_off + l + 64] = d * sc[is + 4] * static_cast<float>(q3);
                dst[base + n_off + l + 96] = d * sc[is + 6] * static_cast<float>(q4);
            }
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

// Reference matmul for Q4_K and Q6_K (dequant-then-FMA). We expose
// these for parity with the Q4_0/Q8_0 paths; the F32 loader doesn't
// actually call them — it just dequantizes into the weight tensor. The
// Q5_K reference is omitted because Q6_K + Q4_K cover the same code
// paths and we don't want to add another one without need.
namespace {

void matmul_q4_K_f32_reference(const uint8_t* qmat, int64_t M, int64_t K,
                                const float* x, float* y) {
    if (K % kQK_K != 0) {
        throw std::runtime_error("matmul_q4_K_f32: K must be divisible by 256");
    }
    std::vector<float> row(static_cast<std::size_t>(K));
    std::size_t bytes_per_row = static_cast<std::size_t>(K / kQK_K) *
                                kQ4_KBlockBytes;
    for (int64_t m = 0; m < M; ++m) {
        dequantize_q4_K(qmat + m * bytes_per_row, K, row.data());
        double acc = 0.0;
        for (int64_t k = 0; k < K; ++k) {
            acc += static_cast<double>(row[static_cast<std::size_t>(k)]) *
                   static_cast<double>(x[k]);
        }
        y[m] = static_cast<float>(acc);
    }
}

void matmul_q6_K_f32_reference(const uint8_t* qmat, int64_t M, int64_t K,
                                const float* x, float* y) {
    if (K % kQK_K != 0) {
        throw std::runtime_error("matmul_q6_K_f32: K must be divisible by 256");
    }
    std::vector<float> row(static_cast<std::size_t>(K));
    std::size_t bytes_per_row = static_cast<std::size_t>(K / kQK_K) *
                                kQ6_KBlockBytes;
    for (int64_t m = 0; m < M; ++m) {
        dequantize_q6_K(qmat + m * bytes_per_row, K, row.data());
        double acc = 0.0;
        for (int64_t k = 0; k < K; ++k) {
            acc += static_cast<double>(row[static_cast<std::size_t>(k)]) *
                   static_cast<double>(x[k]);
        }
        y[m] = static_cast<float>(acc);
    }
}

}  // namespace

void matmul_q4_K_f32(const uint8_t* qmat, int64_t M, int64_t K,
                     const float* x, float* y) {
    // Phase 15: runtime dispatch (same model as Q4_0 above).
#if TINYLLM_ENABLE_AVX2
    if (tinyllm::have_avx2() && tinyllm::have_fma()) {
        tinyllm::ops::matvec_q4_K_f32_avx2(qmat, M, K, x, y);
        return;
    }
#endif
    matmul_q4_K_f32_reference(qmat, M, K, x, y);
}

void matmul_q6_K_f32(const uint8_t* qmat, int64_t M, int64_t K,
                     const float* x, float* y) {
#if TINYLLM_ENABLE_AVX2
    if (tinyllm::have_avx2() && tinyllm::have_fma()) {
        tinyllm::ops::matvec_q6_K_f32_avx2(qmat, M, K, x, y);
        return;
    }
#endif
    matmul_q6_K_f32_reference(qmat, M, K, x, y);
}

}  // namespace tinyllm