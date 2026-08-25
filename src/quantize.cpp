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

#include "tinyllm/matmul.hpp"
#include "tinyllm/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace tinyllm {

// Forward declaration for the AVX2 fused Q4_0 path (see matmul.cpp).
// Defined in tinyllm::ops::matvec_q4_0_f32_avx2; we call it from
// matmul_q4_0_f32 when AVX2 is enabled.
namespace ops {
void matvec_q4_0_f32_avx2(const uint8_t* qmat, int64_t M, int64_t K,
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

#if !TINYLLM_ENABLE_AVX2
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
#endif  // !TINYLLM_ENABLE_AVX2

// (Forward declaration for the AVX2 fused Q4_0 path lives at the top of
// this file, outside this anonymous namespace.)

}  // namespace

void matmul_q4_0_f32(const uint8_t* qmat, int64_t M, int64_t K,
                     const float* x, float* y) {
#if TINYLLM_ENABLE_AVX2
    tinyllm::ops::matvec_q4_0_f32_avx2(qmat, M, K, x, y);
#else
    matmul_q4_0_f32_reference(qmat, M, K, x, y);
#endif
}

void matmul_q8_0_f32(const uint8_t* qmat, int64_t M, int64_t K,
                     const float* x, float* y) {
    matmul_q8_0_f32_reference(qmat, M, K, x, y);
}

}  // namespace tinyllm