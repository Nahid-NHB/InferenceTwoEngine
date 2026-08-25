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

}  // namespace tinyllm::ops