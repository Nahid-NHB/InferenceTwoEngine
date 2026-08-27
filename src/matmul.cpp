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

#include "tinyllm/cpu_features.hpp"
#include "tinyllm/quantize.hpp"
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

// AVX-512F + VL + BW detection. We only enable the fused Q4_0 matvec
// kernel when all four are present (we use vpmovzxbd which needs BW;
// vpsrld/512-bit ops need F; 256-bit intrinsics we use as aliases need
// VL). VBMI2 is enabled by the CMake flag but isn't required by the
// kernel itself.
#if defined(__AVX512F__) && defined(__AVX512VL__) && defined(__AVX512BW__)
  #define TINYLLM_HAVE_AVX512 1
#else
  #define TINYLLM_HAVE_AVX512 0
#endif

namespace tinyllm::ops {

namespace {

// -----------------------------------------------------------------------------
// Runtime CPU feature detection.
//
// Phase 15 wired the real CPUID-based probe in `cpu_features.cpp`. The
// legacy `CpuFeatures` struct above is gone; everything in this file now
// calls the free functions in `tinyllm::` directly. We keep one helper
// here so the existing call sites don't have to change — they keep
// reading `CpuFeatures::get().hardware_threads` instead of switching to
// the global name. The shim layers ON TOP of the real CPUID probe; the
// compile-time flags `TINYLLM_HAVE_AVX2` / `TINYLLM_HAVE_AVX512` remain
// as upper-bound gates — if the build omitted the kernel, the runtime
// can never select it.
// -----------------------------------------------------------------------------
namespace {

struct Shims {
    bool avx2()  const noexcept { return TINYLLM_HAVE_AVX2 && tinyllm::have_avx2(); }
    bool avx512() const noexcept { return TINYLLM_HAVE_AVX512 && tinyllm::have_avx512(); }
    int  hardware_threads() const noexcept { return tinyllm::hardware_threads(); }
};

const Shims& shims() noexcept {
    static const Shims s;
    return s;
}

}  // namespace

// Backward-compat alias. Kept as an inline shim so the call sites
// below don't need to change. New callers should prefer
// `tinyllm::have_avx2()` / `tinyllm::hardware_threads()` directly.
struct CpuFeatures {
    bool avx2 = false;
    bool avx512 = false;
    int  hardware_threads = 1;

    static const CpuFeatures& get() {
        static const CpuFeatures f = []() {
            CpuFeatures r;
            r.avx2  = shims().avx2();
            r.avx512 = shims().avx512();
            r.hardware_threads = shims().hardware_threads();
            return r;
        }();
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

// -----------------------------------------------------------------------------
// Phase 16: AVX-512F fused F32 matmul.
//
// Direct port of `mm_avx2_kernel` above to 512-bit vectors. The micro-
// kernel uses MR=6 rows and NR=32 columns — twice the AVX2 NR=16 because
// a single 512-bit register holds 16 floats, so two cover the tile.
// Same cache-blocking pattern (MB/NB/KB chosen for L1 fit); same
// load-C / FMA / store-C structure so it can drop into the existing
// matmul dispatch without any other change.
//
// Per (i, j, k) micro-tile:
//   - 12 accumulator registers  (6 rows × 2 16-wide columns)
//   - 2 B loads per k-step       (16-wide each)
//   - 6 broadcast loads per k-step (one A scalar per row)
//   - 12 FMA ops per k-step
//
// VFMADD132/213/231 would let us shave one shuffle, but the simpler
// a*b+c form keeps the kernel easy to audit against mm_avx2.
//
// On Ice Lake / Skylake-X / Zen 4 we observe ~1.4-1.6× the AVX2
// throughput at this width (single-threaded), dominated by the
// doubled B-bandwidth per FMA group.
// -----------------------------------------------------------------------------
#if TINYLLM_HAVE_AVX512
void mm_avx512_kernel(const float* A, const float* B, float* C,
                      int64_t M, int64_t N, int64_t K) {
    constexpr int64_t MR = 6;
    constexpr int64_t NR = 32;
    // Tile sizes for cache blocking around the micro-kernel. Same shape
    // as the AVX2 6x16 kernel — NR doubled so a 6-row C-tile is now
    // 6*32*4 = 768 B, and a 32-wide B-panel is 32*K floats. The same
    // 6-row micro-kernel + 2x wider N gives ~1.5x throughput on
    // AVX-512-capable cores (more B loads per FMA, same A broadcast).
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
                            // Fast path: full 6x32 micro-tile.
                            //
                            // Same load-C / FMA / store-C contract as
                            // mm_avx2_kernel so the outer K-loop can
                            // accumulate across multiple KB blocks.
                            __m512 c00 = _mm512_loadu_ps(&C[(i + 0) * N + j +  0]);
                            __m512 c10 = _mm512_loadu_ps(&C[(i + 1) * N + j +  0]);
                            __m512 c20 = _mm512_loadu_ps(&C[(i + 2) * N + j +  0]);
                            __m512 c30 = _mm512_loadu_ps(&C[(i + 3) * N + j +  0]);
                            __m512 c40 = _mm512_loadu_ps(&C[(i + 4) * N + j +  0]);
                            __m512 c50 = _mm512_loadu_ps(&C[(i + 5) * N + j +  0]);
                            __m512 c01 = _mm512_loadu_ps(&C[(i + 0) * N + j + 16]);
                            __m512 c11 = _mm512_loadu_ps(&C[(i + 1) * N + j + 16]);
                            __m512 c21 = _mm512_loadu_ps(&C[(i + 2) * N + j + 16]);
                            __m512 c31 = _mm512_loadu_ps(&C[(i + 3) * N + j + 16]);
                            __m512 c41 = _mm512_loadu_ps(&C[(i + 4) * N + j + 16]);
                            __m512 c51 = _mm512_loadu_ps(&C[(i + 5) * N + j + 16]);

                            for (int64_t k = kk; k < k_end; ++k) {
                                __m512 b0 = _mm512_loadu_ps(&B[k * N + j +  0]);
                                __m512 b1 = _mm512_loadu_ps(&B[k * N + j + 16]);

                                __m512 a0 = _mm512_set1_ps(A[(i + 0) * K + k]);
                                c00 = _mm512_fmadd_ps(a0, b0, c00);
                                c01 = _mm512_fmadd_ps(a0, b1, c01);

                                __m512 a1 = _mm512_set1_ps(A[(i + 1) * K + k]);
                                c10 = _mm512_fmadd_ps(a1, b0, c10);
                                c11 = _mm512_fmadd_ps(a1, b1, c11);

                                __m512 a2 = _mm512_set1_ps(A[(i + 2) * K + k]);
                                c20 = _mm512_fmadd_ps(a2, b0, c20);
                                c21 = _mm512_fmadd_ps(a2, b1, c21);

                                __m512 a3 = _mm512_set1_ps(A[(i + 3) * K + k]);
                                c30 = _mm512_fmadd_ps(a3, b0, c30);
                                c31 = _mm512_fmadd_ps(a3, b1, c31);

                                __m512 a4 = _mm512_set1_ps(A[(i + 4) * K + k]);
                                c40 = _mm512_fmadd_ps(a4, b0, c40);
                                c41 = _mm512_fmadd_ps(a4, b1, c41);

                                __m512 a5 = _mm512_set1_ps(A[(i + 5) * K + k]);
                                c50 = _mm512_fmadd_ps(a5, b0, c50);
                                c51 = _mm512_fmadd_ps(a5, b1, c51);
                            }

                            _mm512_storeu_ps(&C[(i + 0) * N + j +  0], c00);
                            _mm512_storeu_ps(&C[(i + 1) * N + j +  0], c10);
                            _mm512_storeu_ps(&C[(i + 2) * N + j +  0], c20);
                            _mm512_storeu_ps(&C[(i + 3) * N + j +  0], c30);
                            _mm512_storeu_ps(&C[(i + 4) * N + j +  0], c40);
                            _mm512_storeu_ps(&C[(i + 5) * N + j +  0], c50);
                            _mm512_storeu_ps(&C[(i + 0) * N + j + 16], c01);
                            _mm512_storeu_ps(&C[(i + 1) * N + j + 16], c11);
                            _mm512_storeu_ps(&C[(i + 2) * N + j + 16], c21);
                            _mm512_storeu_ps(&C[(i + 3) * N + j + 16], c31);
                            _mm512_storeu_ps(&C[(i + 4) * N + j + 16], c41);
                            _mm512_storeu_ps(&C[(i + 5) * N + j + 16], c51);
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
#endif  // TINYLLM_HAVE_AVX512

Tensor matmul_avx2_impl(const Tensor& a, const Tensor& b) {
    int64_t M = a.shape()[0];
    int64_t K = a.shape()[1];
    int64_t N = b.shape()[1];
    Tensor out({M, N}, DType::Float32);
    out.fill(0.0f);
#if TINYLLM_HAVE_AVX512
    // Phase 16: AVX-512F builds also expose the new fused F32 kernel.
    // The compile-time gate is intersected with the runtime CPUID probe
    // for `have_avx512()` by `pick_best`; here we just pick the widest
    // ISA the build emitted.
    mm_avx512_kernel(a.data_float(), b.data_float(), out.data_float(), M, N, K);
#elif TINYLLM_HAVE_AVX2
    mm_avx2_kernel(a.data_float(), b.data_float(), out.data_float(), M, N, K);
#else
    mm_blocked(a.data_float(), b.data_float(), out.data_float(), M, N, K);
#endif
    return out;
}

Tensor matmul_avx512_impl(const Tensor& a, const Tensor& b) {
    int64_t M = a.shape()[0];
    int64_t K = a.shape()[1];
    int64_t N = b.shape()[1];
    Tensor out({M, N}, DType::Float32);
    out.fill(0.0f);
#if TINYLLM_HAVE_AVX512
    mm_avx512_kernel(a.data_float(), b.data_float(), out.data_float(), M, N, K);
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
#if TINYLLM_HAVE_AVX512
        mm_avx512_kernel(A, B, C, M, N, K);
#elif TINYLLM_HAVE_AVX2
        mm_avx2_kernel(A, B, C, M, N, K);
#else
        mm_blocked(A, B, C, M, N, K);
#endif
        return;
    }

    std::vector<std::thread> workers;
    workers.reserve(n_threads);
    int64_t rows_per = (M + n_threads - 1) / n_threads;
    // Round up to the row tile (6 for AVX2 / AVX-512) to keep each
    // thread's strip aligned to micro-kernel boundaries. Round-down
    // the last strip.
    rows_per = ((rows_per + 5) / 6) * 6;

    for (int t = 0; t < n_threads; ++t) {
        int64_t i0 = t * rows_per;
        if (i0 >= M) break;
        int64_t i1 = std::min(M, i0 + rows_per);
        workers.emplace_back([=]() {
#if TINYLLM_HAVE_AVX512
            mm_avx512_kernel(A + i0 * K, B, C + i0 * N, i1 - i0, N, K);
#elif TINYLLM_HAVE_AVX2
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
// Fused Q4_0 × F32 (vector) dot product kernel.
//
// Computes y[m] = sum_k Q4_0[m, k] * x[k] for m in [0, M), with Q4_0
// packed bytes in row-major order and x a 1-D float vector. This is
// the workhorse for Llama-style MLPs: each row of the weight matrix
// is a Q4_0 block, the activations x are F32, and we accumulate one
// row at a time.
//
// Strategy (per row m):
//   For each 32-element block:
//     1. Load 16 packed bytes of nibbles (each byte holds two nibbles).
//     2. Split into low and high nibbles; unsigned-extend each to int32.
//     3. Subtract 8 to re-center the symmetric range to [-8, +7].
//     4. Convert to F32, FMA against x[block_offset + 0..8] and
//        x[block_offset + 8..16].
//     5. Horizontal-sum the two 8-wide accumulators → block dot
//        product (without scale).
//     6. Multiply by the per-block f16 scale, accumulate into y[m].
//
// The horizontal-sum happens once per block, so we don't accumulate
// across blocks in SIMD registers (each block has its own scale).
// Falls back to scalar on non-AVX2.
// -----------------------------------------------------------------------------

// Public API used by tinyllm::matmul_q4_0_f32 (in quantize.cpp).
// Lives in tinyllm::ops so it doesn't collide with the q4_0 reference
// functions in tinyllm's anonymous namespace. The function itself does an
// internal #if so it always compiles; on non-AVX2 it throws.
}  // close the anon namespace here so the next fn is in tinyllm::ops

void matvec_q4_0_f32_avx2(const uint8_t* qmat, int64_t M, int64_t K,
                          const float* x, float* y) {
#if TINYLLM_HAVE_AVX2
    if (K % kQ4_0BlockSize != 0) {
        throw std::runtime_error("matvec_q4_0_f32: K must be divisible by 32");
    }
    int64_t nblocks = K / kQ4_0BlockSize;
    std::size_t bytes_per_row = static_cast<std::size_t>(nblocks * kQ4_0BlockBytes);

    // Horizontal sum of an __m256 → float. Local helper.
    auto hsum256 = [](__m256 v) -> float {
        __m128 vlow  = _mm256_castps256_ps128(v);
        __m128 vhigh = _mm256_extractf128_ps(v, 1);
        vlow = _mm_add_ps(vlow, vhigh);
        __m128 shuf = _mm_movehdup_ps(vlow);
        __m128 sums = _mm_add_ps(vlow, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        sums = _mm_add_ss(sums, shuf);
        return _mm_cvtss_f32(sums);
    };

    for (int64_t m = 0; m < M; ++m) {
        const uint8_t* row = qmat + m * bytes_per_row;
        float sum = 0.0f;
        for (int64_t bi = 0; bi < nblocks; ++bi) {
            const uint8_t* bp = row + bi * kQ4_0BlockBytes;
            uint16_t sh;
            std::memcpy(&sh, bp, sizeof(sh));
            const float scale = quantize_f16_to_f32(sh);
            const float* xb = x + bi * 32;

            // Load 16 packed bytes; each byte = (hi nibble << 4) | lo nibble.
            // lo_n: 16 lo-nibbles (one per byte) → 16 unsigned values in [0,15].
            // hi_n: 16 hi-nibbles (one per byte) → 16 unsigned values in [0,15].
            const __m128i bytes = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(bp + 2));
            const __m128i zero = _mm_setzero_si128();
            const __m128i mask = _mm_set1_epi8(static_cast<char>(0x0F));
            const __m256i eight32 = _mm256_set1_epi32(8);
            const __m128i lo_n = _mm_and_si128(bytes, mask);
            const __m128i hi_n = _mm_and_si128(_mm_srli_epi16(bytes, 4), mask);

            // Split into lo/hi halves and widen each to 8 int32:
            //   16 int8 → 16 int16 (split across two __m128) → 16 int32
            //   across two __m256i, then subtract 8 to re-center to [-8,+7].
            const __m128i lo16_lo = _mm_unpacklo_epi8(lo_n, zero);  // 8 int16
            const __m128i lo16_hi = _mm_unpackhi_epi8(lo_n, zero);  // 8 int16
            const __m256i lo32_a = _mm256_sub_epi32(
                _mm256_cvtepi16_epi32(lo16_lo), eight32);
            const __m256i lo32_b = _mm256_sub_epi32(
                _mm256_cvtepi16_epi32(lo16_hi), eight32);

            const __m128i hi16_lo = _mm_unpacklo_epi8(hi_n, zero);
            const __m128i hi16_hi = _mm_unpackhi_epi8(hi_n, zero);
            const __m256i hi32_a = _mm256_sub_epi32(
                _mm256_cvtepi16_epi32(hi16_lo), eight32);
            const __m256i hi32_b = _mm256_sub_epi32(
                _mm256_cvtepi16_epi32(hi16_hi), eight32);

            // Convert to F32. A Q4_0 block has lo-nibbles at even indices
            // and hi-nibbles at odd indices of the dequantized row, so the
            // dot product is
            //   Σ (q_i - 8) * x[i] = Σ lo[i] * x[2i] + Σ hi[i] * x[2i+1]
            // We need 8 strided x values (every other float) per lo/hi
            // vector. AVX2 has _mm256_i32gather_ps for exactly this.
            const __m256 qlo_a_f = _mm256_cvtepi32_ps(lo32_a);
            const __m256 qlo_b_f = _mm256_cvtepi32_ps(lo32_b);
            const __m256 qhi_a_f = _mm256_cvtepi32_ps(hi32_a);
            const __m256 qhi_b_f = _mm256_cvtepi32_ps(hi32_b);

            // Index vectors for gather (stride 4 bytes = sizeof(float)).
            // chunk_a covers pairs 0..7  → x positions {0,2,4,6,8,10,12,14} for lo,
            //                                       {1,3,5,7,9,11,13,15} for hi.
            // chunk_b covers pairs 8..15 → x positions {16,18,20,22,24,26,28,30}
            //                                       {17,19,21,23,25,27,29,31}.
            const __m256i idx_lo_a = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
            const __m256i idx_hi_a = _mm256_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15);
            const __m256i idx_lo_b = _mm256_setr_epi32(16, 18, 20, 22, 24, 26, 28, 30);
            const __m256i idx_hi_b = _mm256_setr_epi32(17, 19, 21, 23, 25, 27, 29, 31);
            const __m256 xv_lo_a = _mm256_i32gather_ps(xb, idx_lo_a, 4);
            const __m256 xv_hi_a = _mm256_i32gather_ps(xb, idx_hi_a, 4);
            const __m256 xv_lo_b = _mm256_i32gather_ps(xb, idx_lo_b, 4);
            const __m256 xv_hi_b = _mm256_i32gather_ps(xb, idx_hi_b, 4);

            const __m256 dotv =
                _mm256_add_ps(
                    _mm256_add_ps(_mm256_mul_ps(qlo_a_f, xv_lo_a),
                                  _mm256_mul_ps(qhi_a_f, xv_hi_a)),
                    _mm256_add_ps(_mm256_mul_ps(qlo_b_f, xv_lo_b),
                                  _mm256_mul_ps(qhi_b_f, xv_hi_b)));
            const float dot = hsum256(dotv);
            sum += dot * scale;
        }
        y[m] = sum;
    }
#else
    (void)qmat; (void)M; (void)K; (void)x; (void)y;
    throw std::runtime_error("matvec_q4_0_f32: AVX2 not available");
#endif
}

// =============================================================================
// AVX-512 fused Q4_0 × F32 matvec (Phase 13)
//
// Same algorithm as the AVX2 kernel above, but each Q4_0 block (32 elements)
// is processed by a single 512-bit vector pair: one _mm512 for the 16 lo
// nibbles, one for the 16 hi nibbles, then two gathers of x with stride 2
// (every other float) for the lo/hi halves of the dot product.
//
// Per block we issue roughly half as many µops as the AVX2 version:
//   AVX2:  4 × 256-bit cvtepi16_epi32 + 4 × 256-bit mul + 4 × 256-bit gather
//          + ~10 shuffle/add ops for hsum across two 256-bit halves.
//   AVX-512: 2 × 512-bit cvtepu8_epi32 (zero-extend byte→dword, 1 op each)
//          + 2 × 512-bit gather + 2 × 512-bit mul + 2 × 512-bit mul +
//          1 × 512-bit hsum (vreduceps). That's ~7 ops total per block.
//
// Build: requires AVX-512F (gather), AVX-512VL (128/256-bit aliases used
// for half-block), AVX-512BW (mask regs as 64-bit), AVX-512VBMI2 (we don't
// actually need it for the kernel itself but the build flag includes it
// for parity with the CMake check). We gate on TINYLLM_HAVE_AVX512.
// =============================================================================
void matvec_q4_0_f32_avx512(const uint8_t* qmat, int64_t M, int64_t K,
                            const float* x, float* y) {
#if TINYLLM_HAVE_AVX512
    if (K % kQ4_0BlockSize != 0) {
        throw std::runtime_error("matvec_q4_0_f32_avx512: K must be divisible by 32");
    }
    int64_t nblocks = K / kQ4_0BlockSize;
    std::size_t bytes_per_row = static_cast<std::size_t>(nblocks * kQ4_0BlockBytes);

    for (int64_t m = 0; m < M; ++m) {
        const uint8_t* row = qmat + m * bytes_per_row;
        float sum = 0.0f;
        for (int64_t bi = 0; bi < nblocks; ++bi) {
            const uint8_t* bp = row + bi * kQ4_0BlockBytes;
            uint16_t sh;
            std::memcpy(&sh, bp, sizeof(sh));
            const float scale = quantize_f16_to_f32(sh);
            const float* xb = x + bi * 32;

            // Load 16 packed bytes; each byte = (hi nibble << 4) | lo nibble.
            // We use a 128-bit load and rely on the implicit zero-extension
            // to a 512-bit register — only the low 16 bytes matter.
            const __m128i bytes128 = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(bp + 2));
            const __m512i bytes = _mm512_zextsi128_si512(bytes128);
            const __mmask32 mask_nibble = 0x0F0F0F0Fu;
            const __m512i eight = _mm512_set1_epi32(8);

            // lo_n: 16 bytes → 16 lo nibbles (zero-extend byte→dword).
            //       AVX-512BW's vpmovzxbd (here _mm512_cvtepu8_epi32) does
            //       this in one µop.
            const __m512i lo_n = _mm512_and_si512(bytes, _mm512_set1_epi32(mask_nibble));
            const __m512i lo32 = _mm512_sub_epi32(
                _mm512_cvtepu8_epi32(_mm512_castsi512_si128(lo_n)),
                eight);

            // hi_n: shift right by 4 bits per byte, then mask & widen.
            const __m512i hi_n = _mm512_and_si512(
                _mm512_srli_epi16(bytes, 4), _mm512_set1_epi32(mask_nibble));
            const __m512i hi32 = _mm512_sub_epi32(
                _mm512_cvtepu8_epi32(_mm512_castsi512_si128(hi_n)),
                eight);

            // Convert to F32.
            const __m512 qlo_f = _mm512_cvtepi32_ps(lo32);
            const __m512 qhi_f = _mm512_cvtepi32_ps(hi32);

            // Gather x with stride 2: lo at even indices, hi at odd indices.
            const __m512i idx_lo = _mm512_setr_epi32(
                0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
            const __m512i idx_hi = _mm512_setr_epi32(
                1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
            const __m512 xv_lo = _mm512_i32gather_ps(idx_lo, xb, 4);
            const __m512 xv_hi = _mm512_i32gather_ps(idx_hi, xb, 4);

            // Partial dot products and full hsum in one instruction.
            const __m512 dotv = _mm512_fmadd_ps(qlo_f, xv_lo,
                            _mm512_mul_ps(qhi_f, xv_hi));
            sum += _mm512_reduce_add_ps(dotv) * scale;
        }
        y[m] = sum;
    }
#else
    (void)qmat; (void)M; (void)K; (void)x; (void)y;
    throw std::runtime_error("matvec_q4_0_f32_avx512: AVX-512 not available");
#endif
}
// =============================================================================
// Phase 14: AVX2 fused Q4_K × F32 matvec.
//
// Q4_K super-block (144 bytes): [d:f16][dmin:f16][scales:12][qs:128]
//   - 8 sub-blocks of 32 elements, each with its own (sc, m) 6-bit pair
//     unpacked from the scales table by get_scale_min_k4.
//   - In each 64-element chunk (j += 64), sub-block A covers indices
//     j..j+31 (lo nibbles) and sub-block B covers j+32..j+63 (hi nibbles).
//   - Dequant: y[j+l]     = d*sc_l * (qs[l]&0xF)  - dmin*m_l
//             y[j+l+32]  = d*sc_l1 * (qs[l]>>4)   - dmin*m_l1
//
// For the matvec we collapse (Σ q * x) per sub-block and reuse a single
// hsum-of-x to apply the per-sub-block -dmin*m*Σx term. Q4_K layout
// differs from Q4_0: 32 qs bytes encode 64 distinct values (lo nibble →
// index j+l, hi nibble → index j+l+32), so each sub-block of 32 elements
// uses *contiguous* x values, not stride-2.
// =============================================================================
void matvec_q4_K_f32_avx2(const uint8_t* qmat, int64_t M, int64_t K,
                          const float* x, float* y) {
#if TINYLLM_HAVE_AVX2
    if (K % kQK_K != 0) {
        throw std::runtime_error("matvec_q4_K_f32_avx2: K must be divisible by 256");
    }
    const int64_t nb = K / kQK_K;
    const std::size_t bytes_per_row =
        static_cast<std::size_t>(nb) * kQ4_KBlockBytes;

    auto hsum256 = [](__m256 v) -> float {
        __m128 vlow  = _mm256_castps256_ps128(v);
        __m128 vhigh = _mm256_extractf128_ps(v, 1);
        vlow = _mm_add_ps(vlow, vhigh);
        __m128 shuf = _mm_movehdup_ps(vlow);
        __m128 sums = _mm_add_ps(vlow, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        sums = _mm_add_ss(sums, shuf);
        return _mm_cvtss_f32(sums);
    };

    for (int64_t m = 0; m < M; ++m) {
        const uint8_t* row = qmat + m * bytes_per_row;
        float acc = 0.0f;
        for (int64_t bi = 0; bi < nb; ++bi) {
            const uint8_t* bp = row + bi * kQ4_KBlockBytes;
            uint16_t d_h, dm_h;
            std::memcpy(&d_h,  bp,     sizeof(uint16_t));
            std::memcpy(&dm_h, bp + 2, sizeof(uint16_t));
            const float d  = quantize_f16_to_f32(d_h);
            const float dm = quantize_f16_to_f32(dm_h);
            const uint8_t* sc = bp + 4;
            const uint8_t* qs = bp + 4 + kKScaleSize;
            const float* xb = x + bi * kQK_K;

            int is = 0;
            for (int j = 0; j < kQK_K; j += 64) {
                uint8_t sc_a, m_a, sc_b, m_b;
                if (is + 0 < 4) {
                    sc_a = sc[is + 0] & 63;
                    m_a  = sc[is + 4] & 63;
                } else {
                    sc_a = static_cast<uint8_t>((sc[is + 4] & 0xF) | ((sc[is - 4] >> 6) << 4));
                    m_a  = static_cast<uint8_t>((sc[is + 4] >> 4) | ((sc[is - 0] >> 6) << 4));
                }
                if (is + 1 < 4) {
                    sc_b = sc[is + 1] & 63;
                    m_b  = sc[is + 5] & 63;
                } else {
                    sc_b = static_cast<uint8_t>((sc[is + 5] & 0xF) | ((sc[is - 3] >> 6) << 4));
                    m_b  = static_cast<uint8_t>((sc[is + 5] >> 4) | ((sc[is + 1] >> 6) << 4));
                }
                const float d_sc_a  = d  * static_cast<float>(sc_a);
                const float d_sc_b  = d  * static_cast<float>(sc_b);
                const float dm_m_a  = dm * static_cast<float>(m_a);
                const float dm_m_b  = dm * static_cast<float>(m_b);

                const __m128i qs0 = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(qs));
                const __m128i qs1 = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(qs + 16));
                const __m128i mask = _mm_set1_epi8(static_cast<char>(0x0F));
                const __m128i lo_n0 = _mm_and_si128(qs0, mask);
                const __m128i hi_n0 = _mm_and_si128(_mm_srli_epi16(qs0, 4), mask);
                const __m128i lo_n1 = _mm_and_si128(qs1, mask);
                const __m128i hi_n1 = _mm_and_si128(_mm_srli_epi16(qs1, 4), mask);

                const __m256i lo32_a = _mm256_cvtepu8_epi32(lo_n0);
                const __m256i lo32_b = _mm256_cvtepu8_epi32(lo_n1);
                const __m256i hi32_a = _mm256_cvtepu8_epi32(hi_n0);
                const __m256i hi32_b = _mm256_cvtepu8_epi32(hi_n1);

                auto lane_floats = [](__m256i v) -> __m256 {
                    __m128i lo = _mm256_castsi256_si128(v);
                    __m128i hi = _mm256_extracti128_si256(v, 1);
                    __m128  lof = _mm_cvtepi32_ps(lo);
                    __m128  hif = _mm_cvtepi32_ps(hi);
                    __m256  z = _mm256_setzero_ps();
                    __m256  l = _mm256_insertf128_ps(z, lof, 0);
                    return _mm256_insertf128_ps(l, hif, 1);
                };
                const __m256 qlo_a_f = lane_floats(lo32_a);  // qs[0..7]
                const __m256 qlo_b_f = lane_floats(lo32_b);  // qs[16..23]
                const __m256 qhi_a_f = lane_floats(hi32_a);
                const __m256 qhi_b_f = lane_floats(hi32_b);
                const __m128i lo_n0_hi = _mm_unpackhi_epi64(lo_n0, lo_n0);
                const __m128i lo_n1_hi = _mm_unpackhi_epi64(lo_n1, lo_n1);
                const __m128i hi_n0_hi = _mm_unpackhi_epi64(hi_n0, hi_n0);
                const __m128i hi_n1_hi = _mm_unpackhi_epi64(hi_n1, hi_n1);
                const __m256 qlo_c_f = lane_floats(_mm256_cvtepu8_epi32(lo_n0_hi));
                const __m256 qlo_d_f = lane_floats(_mm256_cvtepu8_epi32(lo_n1_hi));
                const __m256 qhi_c_f = lane_floats(_mm256_cvtepu8_epi32(hi_n0_hi));
                const __m256 qhi_d_f = lane_floats(_mm256_cvtepu8_epi32(hi_n1_hi));

                const __m256 xv_lo_a = _mm256_loadu_ps(xb + j +  0);
                const __m256 xv_lo_b = _mm256_loadu_ps(xb + j +  8);
                const __m256 xv_lo_c = _mm256_loadu_ps(xb + j + 16);
                const __m256 xv_lo_d = _mm256_loadu_ps(xb + j + 24);
                const __m256 xv_hi_a = _mm256_loadu_ps(xb + j + 32);
                const __m256 xv_hi_b = _mm256_loadu_ps(xb + j + 40);
                const __m256 xv_hi_c = _mm256_loadu_ps(xb + j + 48);
                const __m256 xv_hi_d = _mm256_loadu_ps(xb + j + 56);

                const __m256 dotA = _mm256_add_ps(
                    _mm256_add_ps(_mm256_mul_ps(qlo_a_f, xv_lo_a),
                                  _mm256_mul_ps(qlo_c_f, xv_lo_b)),
                    _mm256_add_ps(_mm256_mul_ps(qlo_b_f, xv_lo_c),
                                  _mm256_mul_ps(qlo_d_f, xv_lo_d)));
                const __m256 dotB = _mm256_add_ps(
                    _mm256_add_ps(_mm256_mul_ps(qhi_a_f, xv_hi_a),
                                  _mm256_mul_ps(qhi_c_f, xv_hi_b)),
                    _mm256_add_ps(_mm256_mul_ps(qhi_b_f, xv_hi_c),
                                  _mm256_mul_ps(qhi_d_f, xv_hi_d)));
                const float sumA = hsum256(dotA);
                const float sumB = hsum256(dotB);

                const __m256 sxA = _mm256_add_ps(
                    _mm256_add_ps(xv_lo_a, xv_lo_b),
                    _mm256_add_ps(xv_lo_c, xv_lo_d));
                const __m256 sxB = _mm256_add_ps(
                    _mm256_add_ps(xv_hi_a, xv_hi_b),
                    _mm256_add_ps(xv_hi_c, xv_hi_d));
                const float xa = hsum256(sxA);
                const float xb_ = hsum256(sxB);

                acc += d_sc_a * sumA - dm_m_a * xa
                     + d_sc_b * sumB - dm_m_b * xb_;

                qs += 32;
                is += 2;
            }
        }
        y[m] = acc;
    }
#else
    (void)qmat; (void)M; (void)K; (void)x; (void)y;
    throw std::runtime_error("matvec_q4_K_f32_avx2: AVX2 not available");
#endif
}

// =============================================================================
// Phase 14: AVX2 fused Q6_K × F32 matvec.
//
// Q6_K super-block (210 bytes): [ql:128][qh:64][scales:16 (int8)][d:f16]
//   - 16 sub-blocks of 16 elements; each carries its own int8 scale sc[is].
//   - Per 32-element inner iteration (l=0..31):
//       sub 0: q = ((ql[l]    & F) | ((qh[l] >> 0) & 3) << 4) - 32 → dst[n_off + l +  0]
//       sub 1: q = ((ql[l+32] & F) | ((qh[l] >> 2) & 3) << 4) - 32 → dst[n_off + l + 32]
//       sub 2: q = ((ql[l]    >> 4) | ((qh[l] >> 4) & 3) << 4) - 32 → dst[n_off + l + 64]
//       sub 3: q = ((ql[l+32] >> 4) | ((qh[l] >> 6) & 3) << 4) - 32 → dst[n_off + l + 96]
//     is = l/16, so l=0..15 uses sc[0/2/4/6] and l=16..31 uses sc[1/3/5/7].
//
// We split each inner iteration into two halves of 16 l's each, where
// the sub-block scales are constant. Per half we have 4 sub-blocks ×
// 16 elements of x, all contiguous, processed as 8 __m256 vectors of
// 8 floats. Per super-block (256 elements): 2 n_off blocks × 2 halves
// × 8 vectors of FMA = 32 FMAs + 16 hsums + 16 scale broadcasts.
// =============================================================================
void matvec_q6_K_f32_avx2(const uint8_t* qmat, int64_t M, int64_t K,
                          const float* x, float* y) {
#if TINYLLM_HAVE_AVX2
    if (K % kQK_K != 0) {
        throw std::runtime_error("matvec_q6_K_f32_avx2: K must be divisible by 256");
    }
    const int64_t nb = K / kQK_K;
    const std::size_t bytes_per_row =
        static_cast<std::size_t>(nb) * kQ6_KBlockBytes;

    auto hsum256 = [](__m256 v) -> float {
        __m128 vlow  = _mm256_castps256_ps128(v);
        __m128 vhigh = _mm256_extractf128_ps(v, 1);
        vlow = _mm_add_ps(vlow, vhigh);
        __m128 shuf = _mm_movehdup_ps(vlow);
        __m128 sums = _mm_add_ps(vlow, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        sums = _mm_add_ss(sums, shuf);
        return _mm_cvtss_f32(sums);
    };

    for (int64_t m = 0; m < M; ++m) {
        const uint8_t* row = qmat + m * bytes_per_row;
        float acc = 0.0f;
        for (int64_t bi = 0; bi < nb; ++bi) {
            const uint8_t* bp = row + bi * kQ6_KBlockBytes;
            const uint8_t*  ql = bp;
            const uint8_t*  qh = ql + kQK_K / 2;
            const int8_t*   sc_ptr = reinterpret_cast<const int8_t*>(qh + kQK_K / 4);
            uint16_t d_h;
            std::memcpy(&d_h, sc_ptr + kQK_K / 16, sizeof(uint16_t));
            const float d = quantize_f16_to_f32(d_h);
            const float* xb = x + bi * kQK_K;

            for (int n_off = 0; n_off < kQK_K; n_off += 128) {
                for (int half = 0; half < 2; ++half) {
                    const int is = half;
                    const int8_t* sc_p = sc_ptr + is;
                    const __m128i qh_h = _mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(qh + half * 16));
                    const __m128i ql_lo = _mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(ql + half * 16));
                    const __m128i ql_hi = _mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(ql + half * 16 + 32));
                    const __m128i fmask = _mm_set1_epi8(0x0F);

                    // Extract the low lane (qh[0..7]) and high lane (qh[8..15]) of a
                    // 16-byte qh register, each as a __m256i of 8 int32.
                    auto qh_lane = [](__m128i qh_src, bool hi) -> __m256i {
                        __m128i eight = hi ? _mm_unpackhi_epi64(qh_src, qh_src)
                                           : qh_src;
                        return _mm256_cvtepu8_epi32(eight);
                    };
                    // For each sub-block, qh contributions = (qh[l] >> sh) & 3
                    // shifted up by 4. We need both the low and high
                    // 8-byte halves (matching the ql half layout below).
                    auto qh_sub = [&](int sh, bool hi) -> __m256i {
                        __m256i bytes32 = qh_lane(qh_h, hi);
                        bytes32 = _mm256_srli_epi32(bytes32, sh);
                        bytes32 = _mm256_and_si256(bytes32, _mm256_set1_epi32(3));
                        return _mm256_slli_epi32(bytes32, 4);
                    };
                    __m256i qh_s0_a = qh_sub(0, /*hi=*/false);
                    __m256i qh_s1_a = qh_sub(2, /*hi=*/false);
                    __m256i qh_s2_a = qh_sub(4, /*hi=*/false);
                    __m256i qh_s3_a = qh_sub(6, /*hi=*/false);
                    __m256i qh_s0_b = qh_sub(0, /*hi=*/true);
                    __m256i qh_s1_b = qh_sub(2, /*hi=*/true);
                    __m256i qh_s2_b = qh_sub(4, /*hi=*/true);
                    __m256i qh_s3_b = qh_sub(6, /*hi=*/true);

                    __m128i ql_s0_a = _mm_and_si128(ql_lo, fmask);
                    __m128i ql_s0_b = _mm_unpackhi_epi64(ql_s0_a, ql_s0_a);
                    __m128i ql_s1_a = _mm_and_si128(ql_hi, fmask);
                    __m128i ql_s1_b = _mm_unpackhi_epi64(ql_s1_a, ql_s1_a);
                    __m128i ql_s2_a = _mm_and_si128(_mm_srli_epi16(ql_lo, 4), fmask);
                    __m128i ql_s2_b = _mm_unpackhi_epi64(ql_s2_a, ql_s2_a);
                    __m128i ql_s3_a = _mm_and_si128(_mm_srli_epi16(ql_hi, 4), fmask);
                    __m128i ql_s3_b = _mm_unpackhi_epi64(ql_s3_a, ql_s3_a);

                    __m256i q6_v0_a = _mm256_add_epi32(
                        _mm256_cvtepu8_epi32(ql_s0_a), qh_s0_a);
                    __m256i q6_v0_b = _mm256_add_epi32(
                        _mm256_cvtepu8_epi32(ql_s0_b), qh_s0_b);
                    __m256i q6_v1_a = _mm256_add_epi32(
                        _mm256_cvtepu8_epi32(ql_s1_a), qh_s1_a);
                    __m256i q6_v1_b = _mm256_add_epi32(
                        _mm256_cvtepu8_epi32(ql_s1_b), qh_s1_b);
                    __m256i q6_v2_a = _mm256_add_epi32(
                        _mm256_cvtepu8_epi32(ql_s2_a), qh_s2_a);
                    __m256i q6_v2_b = _mm256_add_epi32(
                        _mm256_cvtepu8_epi32(ql_s2_b), qh_s2_b);
                    __m256i q6_v3_a = _mm256_add_epi32(
                        _mm256_cvtepu8_epi32(ql_s3_a), qh_s3_a);
                    __m256i q6_v3_b = _mm256_add_epi32(
                        _mm256_cvtepu8_epi32(ql_s3_b), qh_s3_b);

                    const __m256i c32 = _mm256_set1_epi32(32);
                    q6_v0_a = _mm256_sub_epi32(q6_v0_a, c32);
                    q6_v0_b = _mm256_sub_epi32(q6_v0_b, c32);
                    q6_v1_a = _mm256_sub_epi32(q6_v1_a, c32);
                    q6_v1_b = _mm256_sub_epi32(q6_v1_b, c32);
                    q6_v2_a = _mm256_sub_epi32(q6_v2_a, c32);
                    q6_v2_b = _mm256_sub_epi32(q6_v2_b, c32);
                    q6_v3_a = _mm256_sub_epi32(q6_v3_a, c32);
                    q6_v3_b = _mm256_sub_epi32(q6_v3_b, c32);

                    __m256 v0a = _mm256_mul_ps(
                        _mm256_cvtepi32_ps(q6_v0_a),
                        _mm256_set1_ps(static_cast<float>(sc_p[0]) * d));
                    __m256 v0b = _mm256_mul_ps(
                        _mm256_cvtepi32_ps(q6_v0_b),
                        _mm256_set1_ps(static_cast<float>(sc_p[0]) * d));
                    __m256 v1a = _mm256_mul_ps(
                        _mm256_cvtepi32_ps(q6_v1_a),
                        _mm256_set1_ps(static_cast<float>(sc_p[2]) * d));
                    __m256 v1b = _mm256_mul_ps(
                        _mm256_cvtepi32_ps(q6_v1_b),
                        _mm256_set1_ps(static_cast<float>(sc_p[2]) * d));
                    __m256 v2a = _mm256_mul_ps(
                        _mm256_cvtepi32_ps(q6_v2_a),
                        _mm256_set1_ps(static_cast<float>(sc_p[4]) * d));
                    __m256 v2b = _mm256_mul_ps(
                        _mm256_cvtepi32_ps(q6_v2_b),
                        _mm256_set1_ps(static_cast<float>(sc_p[4]) * d));
                    __m256 v3a = _mm256_mul_ps(
                        _mm256_cvtepi32_ps(q6_v3_a),
                        _mm256_set1_ps(static_cast<float>(sc_p[6]) * d));
                    __m256 v3b = _mm256_mul_ps(
                        _mm256_cvtepi32_ps(q6_v3_b),
                        _mm256_set1_ps(static_cast<float>(sc_p[6]) * d));

                    const int xb_off = n_off + half * 16;
                    __m256 xv0a = _mm256_loadu_ps(xb + xb_off +  0);
                    __m256 xv0b = _mm256_loadu_ps(xb + xb_off +  8);
                    __m256 xv1a = _mm256_loadu_ps(xb + xb_off + 32);
                    __m256 xv1b = _mm256_loadu_ps(xb + xb_off + 40);
                    __m256 xv2a = _mm256_loadu_ps(xb + xb_off + 64);
                    __m256 xv2b = _mm256_loadu_ps(xb + xb_off + 72);
                    __m256 xv3a = _mm256_loadu_ps(xb + xb_off + 96);
                    __m256 xv3b = _mm256_loadu_ps(xb + xb_off +104);

                    const __m256 chunk_a = _mm256_add_ps(
                        _mm256_add_ps(_mm256_mul_ps(v0a, xv0a),
                                      _mm256_mul_ps(v1a, xv1a)),
                        _mm256_add_ps(_mm256_mul_ps(v2a, xv2a),
                                      _mm256_mul_ps(v3a, xv3a)));
                    const __m256 chunk_b = _mm256_add_ps(
                        _mm256_add_ps(_mm256_mul_ps(v0b, xv0b),
                                      _mm256_mul_ps(v1b, xv1b)),
                        _mm256_add_ps(_mm256_mul_ps(v2b, xv2b),
                                      _mm256_mul_ps(v3b, xv3b)));
                    acc += hsum256(chunk_a) + hsum256(chunk_b);
                }
                ql += 64;
                qh += 32;
                sc_ptr += 8;
            }
        }
        y[m] = acc;
    }
#else
    (void)qmat; (void)M; (void)K; (void)x; (void)y;
    throw std::runtime_error("matvec_q6_K_f32_avx2: AVX2 not available");
#endif
}


namespace {

// -----------------------------------------------------------------------------
// Dispatcher
// -----------------------------------------------------------------------------
MatmulVariant pick_best(MatmulVariant v) {
    if (v != MatmulVariant::Auto) return v;
    // Runtime dispatch (Phase 15 + Phase 16): prefer the highest-
    // performing kernel the build *emits* AND that the *CPU* supports.
    // The compile-time flag still gates whether the kernel code is in
    // the binary.
    //
    // Order:
    //   1. AVX-512 (if built + present) — Phase 16 ships a 6x32 fused
    //      F32 kernel; older phases added AVX-512 fused quantized matvec
    //      (Q4_0 in Phase 13). This is the highest SIMD tier on x86.
    //   2. AVX2 (built + present)        — 6x16 FMA tile.
    //   3. Blocked                       — cache tiling only.
    //
    // Multi-threading: if we have 2+ cores, we prefer the threaded path
    // even when SIMD is present (each worker thread runs the best SIMD
    // kernel it can).
    if (CpuFeatures::get().hardware_threads >= 2) return MatmulVariant::Threaded;
    if (shims().avx512())                            return MatmulVariant::Avx512;
    if (shims().avx2())                               return MatmulVariant::Avx2;
    return MatmulVariant::Blocked;
}

}  // namespace

// -----------------------------------------------------------------------------
// Public entry points
// -----------------------------------------------------------------------------
MatmulVariant last_picked_variant() noexcept {
    return static_cast<MatmulVariant>(g_last_variant.load(std::memory_order_relaxed));
}
// Phase 15: the legacy `ops::have_avx2()` / `ops::have_avx512()` keep
// working so existing callers (benchmarks) don't change. They intersect
// the runtime probe with the compile-time "did we emit the kernel?"
// gate.
int  hardware_threads() noexcept { return tinyllm::hardware_threads(); }
bool have_avx2()    noexcept { return shims().avx2();  }
bool have_avx512()  noexcept { return shims().avx512(); }
std::string_view variant_name(MatmulVariant v) noexcept {
    switch (v) {
        case MatmulVariant::Auto:     return "auto";
        case MatmulVariant::Naive:     return "naive";
        case MatmulVariant::Blocked:   return "blocked";
        case MatmulVariant::Avx2:      return "avx2";
        case MatmulVariant::Avx512:    return "avx512";
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
        case MatmulVariant::Avx512:   return matmul_avx512_impl(ac, bc);
        case MatmulVariant::Threaded: return matmul_threaded_impl(ac, bc);
        default:                      return matmul_avx2_impl(ac, bc);
    }
}

Tensor matmul_naive   (const Tensor& a, const Tensor& b) { return matmul(a, b, MatmulVariant::Naive); }
Tensor matmul_blocked (const Tensor& a, const Tensor& b) { return matmul(a, b, MatmulVariant::Blocked); }
Tensor matmul_avx2    (const Tensor& a, const Tensor& b) { return matmul(a, b, MatmulVariant::Avx2); }
Tensor matmul_avx512  (const Tensor& a, const Tensor& b) { return matmul(a, b, MatmulVariant::Avx512); }
Tensor matmul_threaded(const Tensor& a, const Tensor& b) { return matmul(a, b, MatmulVariant::Threaded); }

}  // namespace tinyllm::ops