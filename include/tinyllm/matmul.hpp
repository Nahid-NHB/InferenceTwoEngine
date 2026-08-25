// include/tinyllm/matmul.hpp
// -----------------------------------------------------------------------------
// Matrix multiplication variants.
//
// We expose several implementations behind a single dispatch so the rest of
// the project can ask for the "best available" without caring about the
// internals. Each variant has the same signature so it can be benchmarked
// head-to-head.
//
//   Naive       — baseline; what was in tensor.cpp.
//   Blocked     — cache-blocked on M / N / K for L1/L2 reuse.
//   Avx2        — AVX2 FMA micro-kernel (6 x 16 register tile) inside blocking.
//   Threaded    — split M across std::threads using the best kernel above.
//
// All variants return a contiguous Tensor of shape (M, N).
// -----------------------------------------------------------------------------
#pragma once

#include "tinyllm/tensor.hpp"

#include <cstdint>
#include <string_view>

namespace tinyllm::ops {

enum class MatmulVariant {
    Auto        = 0,   // pick the best available at runtime
    Naive       = 1,   // baseline triple loop
    Blocked     = 2,   // cache blocked (no SIMD)
    Avx2        = 3,   // AVX2 FMA + cache blocking (runtime-detected)
    Threaded    = 4,   // multi-threaded (uses best available kernel)
};

// 2-D x 2-D matrix multiply. Throws on shape mismatch or non-2-D inputs.
Tensor matmul(const Tensor& a, const Tensor& b, MatmulVariant v = MatmulVariant::Auto);

// Variant entry points (callable directly for benchmarking).
Tensor matmul_naive   (const Tensor& a, const Tensor& b);
Tensor matmul_blocked (const Tensor& a, const Tensor& b);
Tensor matmul_avx2    (const Tensor& a, const Tensor& b);
Tensor matmul_threaded(const Tensor& a, const Tensor& b);

// What did dispatch end up picking? (For logging / debugging.)
MatmulVariant last_picked_variant() noexcept;
std::string_view variant_name(MatmulVariant v) noexcept;

// CPU feature summary. Used by the benchmarking suite to print what
// the build actually targets.
int  hardware_threads() noexcept;
bool have_avx2() noexcept;
bool have_avx512() noexcept;

// Direct entry points to the fused Q4_0 × F32 kernels. Useful for the
// benchmarking suite to put a number on the AVX2 vs AVX-512 gap. The
// static dispatch in `matmul_q4_0_f32` (quantize.cpp) is the path the
// engine actually uses; these are direct.
// Both functions throw std::runtime_error if their ISA isn't compiled in.
void matvec_q4_0_f32_avx2(const std::uint8_t* qmat, std::int64_t M,
                          std::int64_t K, const float* x, float* y);
void matvec_q4_0_f32_avx512(const std::uint8_t* qmat, std::int64_t M,
                            std::int64_t K, const float* x, float* y);

}  // namespace tinyllm::ops