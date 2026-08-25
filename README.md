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
|   2   | Matmul: blocking, SIMD, threading    |   ⏳   |
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
./build/bin/run_all_tests   # 23 unit tests, expects 23/23 pass
./build/bin/bench_tensor    # naive matmul baseline numbers
./build/bin/tinyllm         # CLI smoke test (Phase 1)
```

## Layout

```text
tinyllm/
├── CMakeLists.txt
├── include/tinyllm/
│   ├── memory.hpp       # ref-counted aligned storage
│   └── tensor.hpp       # Tensor + ops + broadcasting + views
├── src/
│   ├── memory.cpp
│   ├── tensor.cpp
│   └── main.cpp         # CLI smoke test for now
├── tests/
│   ├── test_helpers.hpp # minimal REQUIRE/REQUIRE_NEAR harness
│   ├── test_tensor.cpp
│   └── test_main.cpp
└── benchmarks/
    └── bench_tensor.cpp
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
without SIMD and >200 GFLOPS with AVX2. Phase 2 will close that gap.

## License

For personal learning; no license declared yet.
