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
|   3   | BPE tokenizer                        |   ✅   |
|   4   | GGUF model loader                    |   ✅   |
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
./build/bin/run_all_tests   # 54 unit tests (23 tensor, 7 matmul, 13 tokenizer, 11 gguf)
./build/bin/bench_tensor    # naive matmul baseline numbers
./build/bin/bench_matmul    # naive / blocked / AVX2 / threaded comparison
./build/bin/bench_tokenizer # BPE encode throughput
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
│   ├── matmul.hpp       # matmul variants + MatmulVariant enum
│   ├── tokenizer.hpp    # BPE tokenizer
│   └── gguf.hpp         # GGUF file parser
├── src/
│   ├── memory.cpp
│   ├── tensor.cpp
│   ├── matmul.cpp       # naive / blocked / AVX2 / threaded
│   ├── tokenizer.cpp
│   ├── gguf.cpp
│   └── main.cpp         # CLI smoke test
├── tests/
│   ├── test_helpers.hpp # minimal REQUIRE/REQUIRE_NEAR harness
│   ├── test_tensor.cpp
│   ├── test_matmul.cpp
│   ├── test_tokenizer.cpp
│   ├── test_gguf.cpp    # round-trip via synthetic writer
│   └── test_main.cpp
└── benchmarks/
    ├── bench_tensor.cpp
    ├── bench_matmul.cpp
    └── bench_tokenizer.cpp
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

## Phase 3 notes — BPE tokenizer

### Algorithm

1. UTF-8 encode the input into bytes.
2. Convert each byte to its display string (single character).
3. Repeatedly find the adjacent pair with the **lowest merge rank** and
   merge it (concatenate the strings).
4. Stop when no adjacent pair appears in the merge table.
5. Look each resulting string up in the vocab to get a token id.

Decoding inverts this: byte tokens emit their single byte directly; other
tokens emit their (UTF-8) display string.

### On-disk format

We accept two simple file types:

- `vocab.json` — JSON array of strings, index in array == token id. First
  256 entries must be the byte tokens (`"<0x00>"`, ..., `"<0xFF>"`).
- `merges.txt` — one merge per line, `"A B"`, in priority order. `#` for
  comments; blank lines ignored.

This format is intentionally trivial to generate from a real
HuggingFace/SentencePiece tokenizer (a 30-line Python script does it; one
will be added under `examples/` later).

### What's tested (13 cases)

- Byte-token validation at load time (rejects malformed vocabs)
- Pure-ASCII encode, no merges apply
- Single merge ("hi" from "h"+"i")
- Chain of merges ("his")
- Merge priority (lower rank wins when both apply)
- ASCII round-trip
- UTF-8 round-trip (café, résumé, naïve — multi-byte UTF-8 preserved)
- BOS/EOS handling
- `token_to_id` / `id_to_token` lookups
- Unknown merged token falls back to `<unk>`

### Numbers

On the toy vocab (~10 kB text, 264 tokens, 2 merges): ~1.77 MB/s encode
speed. A real LLaMA tokenizer (~32 k vocab, ~30 k merges) will be slower
per pair but the loop is still O(n).

## Phase 4 notes — GGUF model loader

### What we parse

`GgufFile::open(path)` reads a GGUF v3 file (v2 also tolerated) and
exposes:

- The metadata KV pairs (architecture name, hyper-params, tokenizer
  type, etc.).
- A `GgufTensorInfo` per tensor (name, shape, GGUF dtype, offset into
  the data section).
- `load_tensor(idx)` materializes a tensor's data into our `Tensor`
  type. F32 is copied verbatim; F16 is expanded to F32 since our op
  kernels only handle Float32 right now. Quantized types (Q4_0, Q4_1,
  Q8_0) parse and report but raise on load — Phase 8 will wire those
  up.

### File layout recap

```text
[ magic "GGUF" ][ version u32 ][ n_tensors u64 ][ n_kv u64 ]
[ ... n_kv KV pairs ... ]
[ ... n_tensors tensor infos ... ]
[ alignment u64 (v3) ]
[ padding to alignment ]
[ ... tensor data, each at its alignment-multiple offset ... ]
```

### Bug worth noting

The first cut treated `data_section_offset_` as "right after the
alignment field", but the field is followed by padding bytes so the
first tensor's data starts at the next alignment multiple. The fix
rounds the offset up. Caught by the `gguf_loads_f32_*` tests — they
loaded zeros because the reader's seek was 8..24 bytes short.

### What's tested (11 cases)

- Header parse (version, n_tensors, n_kv)
- Scalar metadata: string, uint32
- Tensor infos (names, dims, dtype)
- Tensor loads: F32 matrix, F32 vector, F32 scalar, F16 round-trip
- Bad magic rejected
- Unsupported version rejected
- Dtype name lookups

## License

For personal learning; no license declared yet.
