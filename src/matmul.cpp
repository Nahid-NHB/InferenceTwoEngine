// src/matmul.cpp
// -----------------------------------------------------------------------------
// Matrix multiplication variants.
//
// We implement four versions and dispatch through `matmul(a, b, variant)`.
//
//  Naive       — straight triple loop. Floor of correctness + baseline perf.
//  Blocked     — cache blocking on M, N, K. No SIMD. Big win for any n>~64.
//  Avx2        — 6 x 16 register tile inside cache blocking. Compile-time
//                guarded by __AVX2__; runtime-detected so we can fall back.
//  Threaded    — split M across N worker threads, each runs the best kernel.
//
// Design notes
// ------------
//  * Output is always row-major contiguous.
//  * We *always* operate on contiguous inputs. We make a contiguous copy
//    once at the boundary if needed (cheap for typical transformer dims).
//  * All loops assume contiguous layout. The compiler can auto-vectorize
//    Blocked's inner accumulation; for the AVX2 path we hand-vectorize.
// -----------------------------------------------------------------------------

#include "tinyllm/matmul.hpp"

#include "tinyllm/tensor.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

#if defined(__AVX2__)
  #include <immintrin.h>
  #define TINYLLM_HAVE_AVX2 1
#else
  #define TINYLLM_HAVE_AVX2 0
#endif

namespace tinyllm::ops {

namespace {

// -----------------------------------------------------------------------------
// Runtime CPU feature detection. We do this once at first call.
// -----------------------------------------------------------------------------
struct CpuFeatures {
    bool avx2 = false;
    int  hardware_threads = 1;

    static const CpuFeatures& get() {
        static const CpuFeatures f = detect();
        return f;
    }

private:
    static CpuFeatures detect() {
        CpuFeatures f;
        f.hardware_threads = std::max<unsigned>(1, std::thread::hardware_concurrency());
#if TINYLLM_HAVE_AVX2
        // __AVX2__ is a compile-time macro. To runtime-detect, we'd inspect
        // CPUID. For simplicity (and because we only build with -mavx2 when
        // we want it), we just say "AVX2 is present iff compiled with it."
        f.avx2 = true;
#endif
        return f;
    }
};

// Last variant actually used. Atomically updated so the threaded path can
// record what the workers picked.
std::atomic<int> g_last_variant{0};

// -----------------------------------------------------------------------------
// Naive: straight reference. Used both as a baseline and as the fallback when
// other kernels don't apply (e.g. K not a multiple of the AVX2 width).
// -----------------------------------------------------------------------------
Tensor matmul_naive_impl(const Tensor& a, const Tensor& b) {
    int64_t M = a.shape()[0];
    int64_t K = a.shape()[1];
    int64_t N = b.shape()[1];
    const float* A = a.data_float();
    const float* B = b.data_float();
    Tensor out({M, N}, DType::Float32);
    out.fill(0.0f);
    float* C = out.data_float();

    for (int64_t i = 0; i < M; ++i) {
        for (int64_t k = 0; k < K; ++k) {
            float aik = A[i * K + k];
            for (int64_t j = 0; j < N; ++j) {
                C[i * N + j] += aik * B[k * N + j];
            }
        }
    }
    return out;
}

// -----------------------------------------------------------------------------
// Blocked: classic 3-level cache tiling on M, N, K.
// Tile sizes chosen for L1 (~32 KB): MR=64, NR=64, KR=64 →
//   64*64*4B = 16 KB for each of A's tile, B's tile, C's tile. Fits.
//
// We use i-k-j ordering inside the block to:
//   - keep A's tile row-major & streaming (no strides)
//   - keep B's tile column-friendly (j-innermost so adjacent j's are adjacent
//     in memory)
//   - keep C's tile row-major & accumulate in registers across k
// -----------------------------------------------------------------------------
void mm_blocked(const float* A, const float* B, float* C,
                int64_t M, int64_t N, int64_t K) {
    constexpr int64_t MR = 64;
    constexpr int64_t NR = 64;
    constexpr int64_t KR = 64;

    for (int64_t i0 = 0; i0 < M; i0 += MR) {
        int64_t i_end = std::min(i0 + MR, M);
        for (int64_t j0 = 0; j0 < N; j0 += NR) {
            int64_t j_end = std::min(j0 + NR, N);
            for (int64_t k0 = 0; k0 < K; k0 += KR) {
                int64_t k_end = std::min(k0 + KR, K);
                for (int64_t i = i0; i < i_end; ++i) {
                    for (int64_t k = k0; k < k_end; ++k) {
                        float aik = A[i * K + k];
                        for (int64_t j = j0; j < j_end; ++j) {
                            C[i * N + j] += aik * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

Tensor matmul_blocked_impl(const Tensor& a, const Tensor& b) {
    int64_t M = a.shape()[0];
    int64_t K = a.shape()[1];
    int64_t N = b.shape()[1];
    Tensor out({M, N}, DType::Float32);
    out.fill(0.0f);
    mm_blocked(a.data_float(), b.data_float(), out.data_float(), M, N, K);
    return out;
}

// -----------------------------------------------------------------------------
// AVX2 kernel: 6 rows of C x 16 cols of C per iteration, computed by 12
// 256-bit accumulators. Inner-product along K.
//
// Why 6 x 16?
//   - 16 floats = 2 AVX2 vectors per row → easy to unroll by 2.
//   - 6 rows × 16 cols = 96 outputs per micro-tile. With 12 ymm accumulators
//     we need a spare for streaming loads of A/B; 6 rows x 2 ymm = 12.
//
// Per micro-tile:
//   For each k:
//     broadcast A[i,k] (6 times, one per row)
//     load B[k, j..j+16] (2 vectors)
//     FMA into the 12 accumulators
//
// After K, write 12 vectors back to C.
// -----------------------------------------------------------------------------
#if TINYLLM_HAVE_AVX2
void mm_avx2_kernel(const float* A, const float* B, float* C,
                    int64_t M, int64_t N, int64_t K) {
    constexpr int64_t MR = 6;
    constexpr int64_t NR = 16;
    // Tile sizes for cache blocking around the micro-kernel. A 6x64 A-panel,
    // a 64x16 B-panel, and a 6x16 C-tile fit comfortably in L1.
    constexpr int64_t MB = 96;
    constexpr int64_t NB = 256;
    constexpr int64_t KB = 64;

    for (int64_t ii = 0; ii < M; ii += MB) {
        int64_t i_end = std::min(ii + MB, M);
        for (int64_t jj = 0; jj < N; jj += NB) {
            int64_t j_end = std::min(jj + NB, N);
            for (int64_t kk = 0; kk < K; kk += KB) {
                int64_t k_end = std::min(kk + KB, K);
                for (int64_t i = ii; i < i_end; i += MR) {
                    int64_t i_mr = std::min<int64_t>(MR, i_end - i);
                    for (int64_t j = jj; j < j_end; j += NR) {
                        int64_t j_nr = std::min<int64_t>(NR, j_end - j);
                        if (i_mr == MR && j_nr == NR) {
                            // Fast path: full 6x16 micro-tile.
                            //
                            // The kernel runs once per (i_block, j_block, k_block)
                            // triple. For multiple k blocks we must ACCUMULATE into
                            // C, not overwrite it. So we load the current contents of
                            // C into the accumulators, then FMA in this k-block's
                            // contribution, then store back.
                            __m256 c00 = _mm256_loadu_ps(&C[(i + 0) * N + j + 0]);
                            __m256 c01 = _mm256_loadu_ps(&C[(i + 0) * N + j + 8]);
                            __m256 c10 = _mm256_loadu_ps(&C[(i + 1) * N + j + 0]);
                            __m256 c11 = _mm256_loadu_ps(&C[(i + 1) * N + j + 8]);
                            __m256 c20 = _mm256_loadu_ps(&C[(i + 2) * N + j + 0]);
                            __m256 c21 = _mm256_loadu_ps(&C[(i + 2) * N + j + 8]);
                            __m256 c30 = _mm256_loadu_ps(&C[(i + 3) * N + j + 0]);
                            __m256 c31 = _mm256_loadu_ps(&C[(i + 3) * N + j + 8]);
                            __m256 c40 = _mm256_loadu_ps(&C[(i + 4) * N + j + 0]);
                            __m256 c41 = _mm256_loadu_ps(&C[(i + 4) * N + j + 8]);
                            __m256 c50 = _mm256_loadu_ps(&C[(i + 5) * N + j + 0]);
                            __m256 c51 = _mm256_loadu_ps(&C[(i + 5) * N + j + 8]);

                            for (int64_t k = kk; k < k_end; ++k) {
                                __m256 b0 = _mm256_loadu_ps(&B[k * N + j + 0]);
                                __m256 b1 = _mm256_loadu_ps(&B[k * N + j + 8]);

                                __m256 a0 = _mm256_set1_ps(A[i * K + k]);
                                c00 = _mm256_fmadd_ps(a0, b0, c00);
                                c01 = _mm256_fmadd_ps(a0, b1, c01);

                                __m256 a1 = _mm256_set1_ps(A[(i + 1) * K + k]);
                                c10 = _mm256_fmadd_ps(a1, b0, c10);
                                c11 = _mm256_fmadd_ps(a1, b1, c11);

                                __m256 a2 = _mm256_set1_ps(A[(i + 2) * K + k]);
                                c20 = _mm256_fmadd_ps(a2, b0, c20);
                                c21 = _mm256_fmadd_ps(a2, b1, c21);

                                __m256 a3 = _mm256_set1_ps(A[(i + 3) * K + k]);
                                c30 = _mm256_fmadd_ps(a3, b0, c30);
                                c31 = _mm256_fmadd_ps(a3, b1, c31);

                                __m256 a4 = _mm256_set1_ps(A[(i + 4) * K + k]);
                                c40 = _mm256_fmadd_ps(a4, b0, c40);
                                c41 = _mm256_fmadd_ps(a4, b1, c41);

                                __m256 a5 = _mm256_set1_ps(A[(i + 5) * K + k]);
                                c50 = _mm256_fmadd_ps(a5, b0, c50);
                                c51 = _mm256_fmadd_ps(a5, b1, c51);
                            }

                            _mm256_storeu_ps(&C[(i + 0) * N + j + 0], c00);
                            _mm256_storeu_ps(&C[(i + 0) * N + j + 8], c01);
                            _mm256_storeu_ps(&C[(i + 1) * N + j + 0], c10);
                            _mm256_storeu_ps(&C[(i + 1) * N + j + 8], c11);
                            _mm256_storeu_ps(&C[(i + 2) * N + j + 0], c20);
                            _mm256_storeu_ps(&C[(i + 2) * N + j + 8], c21);
                            _mm256_storeu_ps(&C[(i + 3) * N + j + 0], c30);
                            _mm256_storeu_ps(&C[(i + 3) * N + j + 8], c31);
                            _mm256_storeu_ps(&C[(i + 4) * N + j + 0], c40);
                            _mm256_storeu_ps(&C[(i + 4) * N + j + 8], c41);
                            _mm256_storeu_ps(&C[(i + 5) * N + j + 0], c50);
                            _mm256_storeu_ps(&C[(i + 5) * N + j + 8], c51);
                        } else {
                            // Slow tail: edges of M or N. Fall back to scalar.
                            for (int64_t k = k_end; k-- > kk; ) {
                                for (int64_t rr = 0; rr < i_mr; ++rr) {
                                    float aik = A[(i + rr) * K + k];
                                    for (int64_t cc = 0; cc < j_nr; ++cc) {
                                        C[(i + rr) * N + j + cc] += aik * B[k * N + j + cc];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
#endif  // TINYLLM_HAVE_AVX2

Tensor matmul_avx2_impl(const Tensor& a, const Tensor& b) {
    int64_t M = a.shape()[0];
    int64_t K = a.shape()[1];
    int64_t N = b.shape()[1];
    Tensor out({M, N}, DType::Float32);
    out.fill(0.0f);
#if TINYLLM_HAVE_AVX2
    mm_avx2_kernel(a.data_float(), b.data_float(), out.data_float(), M, N, K);
#else
    mm_blocked(a.data_float(), b.data_float(), out.data_float(), M, N, K);
#endif
    return out;
}

// -----------------------------------------------------------------------------
// Threaded: split M across worker threads. Each worker computes a strip of C.
// We use a simple partition over the outermost M-block so threads don't
// fight for the same cache lines.
// -----------------------------------------------------------------------------
void mm_threaded_dispatch(const float* A, const float* B, float* C,
                          int64_t M, int64_t N, int64_t K, int n_threads) {
    n_threads = std::max(1, n_threads);
    n_threads = std::min<int64_t>(n_threads, M / 64);  // don't spawn useless threads
    if (n_threads <= 1) {
#if TINYLLM_HAVE_AVX2
        mm_avx2_kernel(A, B, C, M, N, K);
#else
        mm_blocked(A, B, C, M, N, K);
#endif
        return;
    }

    std::vector<std::thread> workers;
    workers.reserve(n_threads);
    int64_t rows_per = (M + n_threads - 1) / n_threads;
    // Round up to the AVX2 row tile (6) to keep each thread's strip aligned
    // to micro-kernel boundaries. Round-down the last strip.
    rows_per = ((rows_per + 5) / 6) * 6;

    for (int t = 0; t < n_threads; ++t) {
        int64_t i0 = t * rows_per;
        if (i0 >= M) break;
        int64_t i1 = std::min(M, i0 + rows_per);
        workers.emplace_back([=]() {
#if TINYLLM_HAVE_AVX2
            mm_avx2_kernel(A + i0 * K, B, C + i0 * N, i1 - i0, N, K);
#else
            mm_blocked(A + i0 * K, B, C + i0 * N, i1 - i0, N, K);
#endif
        });
    }
    for (auto& th : workers) th.join();
}

Tensor matmul_threaded_impl(const Tensor& a, const Tensor& b) {
    int64_t M = a.shape()[0];
    int64_t K = a.shape()[1];
    int64_t N = b.shape()[1];
    Tensor out({M, N}, DType::Float32);
    out.fill(0.0f);
    mm_threaded_dispatch(a.data_float(), b.data_float(), out.data_float(),
                         M, N, K, CpuFeatures::get().hardware_threads);
    return out;
}

// -----------------------------------------------------------------------------
// Dispatcher
// -----------------------------------------------------------------------------
MatmulVariant pick_best(MatmulVariant v) {
    if (v != MatmulVariant::Auto) return v;
    if (CpuFeatures::get().hardware_threads >= 2) return MatmulVariant::Threaded;
#if TINYLLM_HAVE_AVX2
    return MatmulVariant::Avx2;
#else
    return MatmulVariant::Blocked;
#endif
}

}  // namespace

// -----------------------------------------------------------------------------
// Public entry points
// -----------------------------------------------------------------------------
MatmulVariant last_picked_variant() noexcept {
    return static_cast<MatmulVariant>(g_last_variant.load(std::memory_order_relaxed));
}
std::string_view variant_name(MatmulVariant v) noexcept {
    switch (v) {
        case MatmulVariant::Auto:     return "auto";
        case MatmulVariant::Naive:     return "naive";
        case MatmulVariant::Blocked:   return "blocked";
        case MatmulVariant::Avx2:      return "avx2";
        case MatmulVariant::Threaded:  return "threaded";
    }
    return "unknown";
}

Tensor matmul(const Tensor& a, const Tensor& b, MatmulVariant v) {
    if (a.dtype() != DType::Float32 || b.dtype() != DType::Float32) {
        throw std::invalid_argument("matmul: only float32 supported");
    }
    if (a.shape().size() != 2 || b.shape().size() != 2) {
        throw std::invalid_argument("matmul: inputs must be 2-D");
    }
    if (a.shape()[1] != b.shape()[0]) {
        throw std::invalid_argument("matmul: shape mismatch");
    }
    Tensor ac = a.is_contiguous() ? a : a.contiguous();
    Tensor bc = b.is_contiguous() ? b : b.contiguous();

    MatmulVariant picked = pick_best(v);
    g_last_variant.store(static_cast<int>(picked), std::memory_order_relaxed);

    switch (picked) {
        case MatmulVariant::Naive:    return matmul_naive_impl(ac, bc);
        case MatmulVariant::Blocked:  return matmul_blocked_impl(ac, bc);
        case MatmulVariant::Avx2:     return matmul_avx2_impl(ac, bc);
        case MatmulVariant::Threaded: return matmul_threaded_impl(ac, bc);
        default:                      return matmul_avx2_impl(ac, bc);
    }
}

Tensor matmul_naive   (const Tensor& a, const Tensor& b) { return matmul(a, b, MatmulVariant::Naive); }
Tensor matmul_blocked (const Tensor& a, const Tensor& b) { return matmul(a, b, MatmulVariant::Blocked); }
Tensor matmul_avx2    (const Tensor& a, const Tensor& b) { return matmul(a, b, MatmulVariant::Avx2); }
Tensor matmul_threaded(const Tensor& a, const Tensor& b) { return matmul(a, b, MatmulVariant::Threaded); }

}  // namespace tinyllm::ops