// include/tinyllm/quantize.hpp
// -----------------------------------------------------------------------------
// Block-wise symmetric quantization (Q8_0, Q4_0).
//
// Both formats quantize 32-element blocks of Float32 to:
//   - Q8_0: 1×f16 scale + 32×int8 values   (34 bytes / block)
//   - Q4_0: 1×f16 scale + 16 packed nibbles (18 bytes / block)
//
// Layout (little-endian on disk, matches the GGUF v3 spec):
//
//   Q8_0 block:
//     [f16 scale][int8 qs[0]...qs[31]]
//     dequantized value i:  (qs[i] * scale)
//
//   Q4_0 block:
//     [f16 scale][packed: qs[0] lo nibble, qs[1] hi nibble, ...]
//     where each nibble is in [0, 15] representing a signed value in
//     [-8, 7]. dequantized value i:  ((q - 8) * scale).
//
// We use scales as f16 (the GGUF spec). The half-to-float conversion
// lives in `quantize_f16_to_f32` and is shared with gguf.cpp's F16 path
// but kept independent here for testability.
// -----------------------------------------------------------------------------
#pragma once

#include "tinyllm/tensor.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace tinyllm {

// Block size for Q4_0 / Q8_0. 32 elements per block.
constexpr int64_t kQ4_0BlockSize = 32;
constexpr int64_t kQ8_0BlockSize = 32;

// On-disk sizes per block.
constexpr std::size_t kQ4_0BlockBytes = 2 /*f16 scale*/ + 16 /*nibbles*/;
constexpr std::size_t kQ8_0BlockBytes = 2 /*f16 scale*/ + 32 /*int8*/;

// Half-precision <-> single-precision helpers. Both conversion functions
// return float; they round-half-to-even on the half side (the same as
// IEEE-754 default).
float quantize_f16_to_f32(uint16_t h) noexcept;
uint16_t quantize_f32_to_f16(float x) noexcept;

// -----------------------------------------------------------------------------
// Q8_0: symmetric 8-bit, block size 32.
// -----------------------------------------------------------------------------

// One Q8_0 block in its in-memory form. `qs[i]` is signed [-127, 127].
struct Q8_0Block {
    float                scale;        // block absmax / 127
    std::array<int8_t, kQ8_0BlockSize> qs;  // quantized values
};

// Quantize 32 floats to one Q8_0 block. `src` must have 32 elements.
// If the block is all zeros, scale=0 and qs are all zero.
Q8_0Block quantize_q8_0_block(const float* src) noexcept;

// Dequantize one Q8_0 block back to 32 floats. `dst` must have 32 slots.
void dequantize_q8_0_block(const Q8_0Block& b, float* dst) noexcept;

// Quantize a flat F32 tensor (any shape, total length divisible by 32)
// to a packed Q8_0 byte stream. The returned vector has
// (n/32) * kQ8_0BlockBytes bytes, in block order.
std::vector<uint8_t> quantize_q8_0(const float* src, int64_t n);

// Dequantize a packed Q8_0 byte stream to a flat F32 buffer.
// `n` is the number of F32 values (must equal blocks * 32).
void dequantize_q8_0(const uint8_t* packed, int64_t n, float* dst);

// -----------------------------------------------------------------------------
// Q4_0: symmetric 4-bit, block size 32.
//
// Nibbles are stored in [0, 15] but represent signed values in [-8, 7].
// Stored as (q + 8) so the high bit is 0, simplifying mask logic on
// some hardware. We follow the GGUF convention exactly: qs[i] nibble
// value `n` represents a signed int `(int)n - 8`.
// -----------------------------------------------------------------------------

// One Q4_0 block in its in-memory form.
struct Q4_0Block {
    float                    scale;  // block absmax / 7 (we map to [-8, 7])
    std::array<uint8_t, 16>  qs;     // packed nibbles: 2 values per byte
                                     // (low nibble first in the GGUF spec
                                     // is at index 0, but the exact
                                     // interleave varies across versions;
                                     // we use the standard low=even,
                                     // high=odd convention.)
};

// Quantize 32 floats to one Q4_0 block.
Q4_0Block quantize_q4_0_block(const float* src) noexcept;

// Dequantize one Q4_0 block to 32 floats.
void dequantize_q4_0_block(const Q4_0Block& b, float* dst) noexcept;

// Pack / unpack the entire flat F32 tensor to a Q4_0 byte stream.
// `n` must be divisible by 32.
std::vector<uint8_t> quantize_q4_0(const float* src, int64_t n);
void dequantize_q4_0(const uint8_t* packed, int64_t n, float* dst);

// -----------------------------------------------------------------------------
// Reference matmul on quantized weights: dequantize on the fly and
// accumulate. This is what we'll replace with a fused kernel in Phase 9.
//
//   y[m] = sum_k Q4_0(m, k) * x[k]    (one row of A is Q4_0; B is F32)
//
// For now we dequantize each row into a fresh float buffer and call the
// existing F32 matmul. The shape of `qmat` is `[M, K]` with `K`
// divisible by 32; `x` is `[K]`; `y` is `[M]`. We only handle the Q4_0
// row-major case here.
// -----------------------------------------------------------------------------
void matmul_q4_0_f32(const uint8_t* qmat, int64_t M, int64_t K,
                     const float* x, float* y);
void matmul_q8_0_f32(const uint8_t* qmat, int64_t M, int64_t K,
                     const float* x, float* y);

}  // namespace tinyllm