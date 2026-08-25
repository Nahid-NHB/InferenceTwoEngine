# tinyllm

A tiny LLM inference engine built from scratch in modern C++.

This is a **learning + portfolio project** inspired by `llama.cpp`, `ggml`, and
`tinygrad`. The goal is to understand and implement the core components
ourselves — no PyTorch, no TensorFlow, no ONNX, no llama.cpp.

## Status

| Phase | Topic                                | Status |
|------:|--------------------------------------|:------:|
|   0   | Project architecture & CMake build   |   ✅   |
|   1   | Tensor library (Float32/Int32, views)|   ✅   |
|   2   | Matmul: blocking, SIMD, threading    |   ✅   |
|   3   | BPE tokenizer                        |   ⏳   |
|   4   | GGUF model loader                    |   ⏳   |
|   5   | Llama-style transformer              |   ⏳   |
|   6   | KV cache                             |   ⏳   |
|   7   | Sampling (greedy / top-k / top-p)    |   ⏳   |
|   8   | Quantization (INT8 → INT4)           |   ⏳   |
|   9   | Performance engineering (AVX2/512)   |   ⏳   |
|  10   | Benchmarking suite                   |   ⏳   |

## Build

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Requirements: CMake ≥ 3.20, a C++20 compiler (GCC 11+, Clang 14+), Ninja.

## Test & benchmark

```bash
./build/bin/run_all_tests   # 30 unit tests (23 tensor + 7 matmul)
./build/bin/bench_tensor    # naive matmul baseline numbers
./build/bin/bench_matmul    # naive / blocked / AVX2 / threaded comparison
./build/bin/phase1_demo     # 2-layer MLP smoke test
./build/bin/tinyllm         # CLI smoke test
```

## Layout

```text
tinyllm/
├── CMakeLists.txt
├── include/tinyllm/
│   ├── memory.hpp       # ref-counted aligned storage
│   ├── tensor.hpp       # Tensor + ops + broadcasting + views
│   └── matmul.hpp       # matmul variants + MatmulVariant enum
├── src/
│   ├── memory.cpp
│   ├── tensor.cpp
│   ├── matmul.cpp       # naive / blocked / AVX2 / threaded
│   └── main.cpp         # CLI smoke test
├── tests/
│   ├── test_helpers.hpp # minimal REQUIRE/REQUIRE_NEAR harness
│   ├── test_tensor.cpp
│   ├── test_matmul.cpp  # cross-variant correctness
│   └── test_main.cpp
└── benchmarks/
    ├── bench_tensor.cpp
    └── bench_matmul.cpp # all 4 variants at N ∈ {64…1024}
```

## Phase 1 notes — tensor library

### Design choices

- **Row-major contiguous** layout (last dim is fastest-varying).
- **Element strides, not byte strides**, in the public API. Easier
  broadcasting math, no `sizeof(T)` tax at every call site.
- **Ref-counted storage** via `std::shared_ptr<Storage>` with a stateful
  deleter. Multiple `Tensor`s can share the same buffer (zero-copy views).
- **Single allocation** for the storage header + aligned data. Saves one
  malloc/free per tensor and keeps the header in cache next to the data.
- **64-byte alignment** for all allocations (covers AVX-512 with room to
  spare; same cost as smaller alignments on modern allocators).
- **Element-wise ops** support N-D broadcasting via right-aligned shape
  rules: dimensions of size 1 broadcast, otherwise must match.
- **Naive O(n³) matmul** as the baseline — we will cache-block / SIMD /
  thread it in Phase 2 with benchmarks at every step.

### Numerical correctness

- `softmax` uses the max-subtraction trick — verified by a stability test
  with input `[1000, 1001, 1002]`.
- Reductions accumulate in `double` and cast back to `float` at the end to
  limit catastrophic cancellation on long reductions.
- Element-wise + reductions, matmul, broadcasting, transpose views,
  contiguous copy, reshape, squeeze/unsqueeze — all have unit tests.

### Baseline numbers (1 core, naive matmul)

| size  |  ms   |  GFLOPS  |
|------:|------:|---------:|
|   64  |  0.21 |   2.53   |
|  128  |  2.07 |   2.03   |
|  256  | 18.39 |   1.83   |
|  512  |184.62 |   1.45   |

Plenty of headroom. Modern x86 cores can deliver ~30 GFLOPS single-precision
without SIMD and >200 GFLOPS with AVX2. Phase 2 closes that gap — see below.

## Phase 2 notes — matrix multiplication

### Variants

- **`Naive`** — straight triple loop. Baseline.
- **`Blocked`** — cache-blocked on M, N, K (64³ tile fits in L1).
- **`Avx2`** — 6×16 register micro-kernel inside cache blocking, 12
  accumulators in flight. Each micro-tile reuses one row of B against six
  rows of A.
- **`Threaded`** — splits M across `std::thread`s; each thread runs the
  AVX2 (or blocked) kernel.

### Numbers (release build, AVX2 + FMA enabled, all cores)

N×N matmul, GFLOPS (higher is better):

| size |  naive | blocked |  AVX2 | threaded |
|-----:|-------:|--------:|------:|---------:|
|   64 |   19.2 |    19.0 |  69.6 |     69.6 |
|  128 |   19.0 |    17.0 |  44.7 |     14.0 |
|  256 |   17.8 |    17.1 |  63.6 |     76.4 |
|  512 |   13.2 |    15.0 |  52.7 |    126.1 |
| 1024 |    8.5 |    14.5 |  52.3 |     79.5 |

Observations:

- **SIMD is the single biggest win** — 5–6× over naive at every size.
- **Blocking helps modestly** when the problem is small enough that the
  working set fits in cache; once we're memory-bound, blocking alone
  doesn't help much.
- **Threading** scales well for large N (≥512) where each thread has
  enough work; for small N the overhead of spawning threads hurts.

The "auto" pick for the current CPU is `Threaded`.

### Bug worth noting

The first cut of `mm_avx2_kernel` had a classic mistake: it `store`d the
12 accumulators to C after each `kk` block, **overwriting** the partial sum
that had been computed for previous `kk` blocks. The fix is to `load` the
existing C into the accumulators at the start of each `kk` block, FMA in
this block's contribution, then store back. That bug was caught by the
`variants_agree_large` test in ~3 seconds — writing the test first paid
off.

## License

For personal learning; no license declared yet.
