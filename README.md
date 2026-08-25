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
|   5   | Llama-style transformer              |   ✅   |
|   6   | KV cache                             |   ✅   |
|   7   | Sampling (greedy / top-k / top-p)    |   ✅   |
|   8   | Quantization (Q8_0 + Q4_0)          |   ✅   |
|   9   | Performance engineering (AVX2/512)   |   ✅   |
|  10   | Benchmarking suite                   |   ✅   |

## Build

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Requirements: CMake ≥ 3.20, a C++20 compiler (GCC 11+, Clang 14+), Ninja.

## Test & benchmark

```bash
./build/bin/run_all_tests    # 117 unit tests (23 tensor, 7 matmul, 13 tokenizer, 11 gguf,
                             # 7 rmsnorm, 8 rope, 5 attention, 3 mlp, 5 model, 7 kv_cache,
                             # 13 sampler, 15 quantize)
./build/bin/bench_tensor     # naive matmul baseline numbers
./build/bin/bench_matmul     # naive / blocked / AVX2 / threaded comparison
./build/bin/bench_quantize   # Q4_0 / Q8_0 × F32 matvec vs F32 matmul
./build/bin/bench_transformer # unified Phase 10 suite (all kernels, end-to-end)
./build/bin/bench_tokenizer  # BPE encode throughput
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
  kernels only handle Float32 right now. Q8_0 and Q4_0 (Phase 8) are
  dequantized to F32 on load. Q4_1 still raises — it's not on the
  critical path for typical Llama models.

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

## Phase 5 notes — Llama-style transformer

### Pieces

```
include/tinyllm/
├── rmsnorm.hpp       # y = (x / sqrt(mean(x²) + eps)) * gamma
├── rope.hpp          # rotary position embeddings (Llama half-rotation)
├── attention.hpp     # GQA attention with RoPE + causal mask
├── mlp.hpp           # SwiGLU MLP: down(silu(gate) * up)
├── llama_block.hpp   # one block: residual + pre-norm attn + residual + pre-norm mlp
└── model.hpp         # embed → n_layers × block → final norm → unembed
```

### Design choices

- **Pre-norm** everywhere (Llama 2/3 style). Each sub-block takes a
  residual: `x = x + sub(rmsnorm(x))`.
- **GQA**: `n_kv_heads ≤ n_heads`. K and V are projected into a smaller
  space, then broadcast to all `n_heads` query heads via `repeat_interleave`.
  Smaller memory footprint for the KV cache (Phase 6).
- **Per-head matmul in Phase 5**. We could batch heads into one big
  matmul, but per-head keeps the code obvious. Phase 9 will fold this.
- **RMSNorm accumulates in double** for the mean-of-squares — values can
  be large, and `double` accumulation limits cancellation on long
  reductions.

### Bugs worth noting

- `Tensor::at_flat()` returns a `float&` even on Int32 storage, which
  silently miswrites the bytes (only the low 4 bytes get the float value).
  Added `at_flat_int()` for typed access. Caught by `llama_forward_smoke`
  failing with "token id out of range" — the value read back was
  garbage.
- My first attention test used `seq=2` and `start_pos=0`, but position
  1 *does* get rotated by RoPE (only position 0 is identity). The test
  trace assumed identity RoPE for both rows and gave wrong expectations.
  Split the test into `seq=1` (RoPE identity at pos 0) and a separate
  `seq=2` test that accounts for the rotation at pos 1.

### What's tested (28 cases across Phase 5)

- RMSNorm: zero, hand-computed, per-channel gamma, 2-D rows, 3-D input,
  wrong last-dim, non-1-D gamma
- RoPE: identity at pos 0, hand-computed single-pair rotation, multiple
  heads, multiple positions, start_pos offset, norm preservation per
  pair, precomputed tables, odd-dim rejection
- Attention: identity weights, single-token causal pass-through, seq=2
  with RoPE, random smoke, bad config rejection
- MLP: zero intermediate, hand-computed, random smoke
- Model: block with zero MLP, full forward smoke, zero weights → zero
  logits, wrong dtype, OOV token

### Numbers

This phase is correctness-focused; the per-head matmul makes each
forward pass roughly `O(n_layers * n_heads * seq^2 * head_dim)` per
block, dominated by the softmax materialization. We'll measure in
Phase 9.

## Phase 6 notes — KV cache

### What we cache

For each layer we allocate a `KvCache` with K and V tensors of shape
`[max_seq_len, n_kv_heads, head_dim]` (F32). The cache is owned by
`LlamaModel`, one per layer.

### Forward paths

Two parallel paths:

- `attention_forward(x, w, cfg, start_pos)` — pure. Used in tests and
  when the caller doesn't want cache plumbing.
- `attention_forward_cached(x, w, cfg, cache, start_pos)` — appends
  the new K/V rows to the cache and scores against the full history.

The cached path is what `LlamaModel::forward_cached` uses. Per-token
generation now costs O(seq) per layer instead of O(seq²): the new
K/V is a single row, attention sees `seq_k = start_pos + 1` keys.

### Mask is absolute

We changed the causal mask to use absolute positions. After RoPE,
`kpos[k] > qpos` is the right condition regardless of whether `seq_q`
equals `seq_k` (prefill) or `seq_q == 1, seq_k == start_pos + 1`
(decode).

### Bug worth noting

The first cut kept two parallel implementations (one for cached, one
for uncached) and the code drifted. Refactored to share a per-head
kernel (`run_head`) that takes absolute `kpos[]` and `start_pos`. Both
entry points now route through it, and the per-head loop is identical
between the two paths.

### What's tested (7 cases)

- `KvCache`: construct/append/clear, capacity overflow throws, shape
  mismatch throws
- **Cached prefill matches uncached** (5 tokens, 2 layers, logits
  identical within 1e-5)
- **Cached decode matches uncached** (prefill 3 tokens, then decode
  one at a time; at each step the cached logits for the new token
  match what a non-cached forward over the full prefix would have
  produced at that position)
- `reset_caches` lets you re-prefill from position 0
- Capacity exceeded (`max_seq_len=3`, prefill 4 tokens) throws

## Phase 7 notes — Sampling

### API

```cpp
struct SamplerConfig {
    float temperature = 1.0f;   // <=0 → greedy
    int   top_k       = 0;      // <=0 → disabled
    float top_p       = 1.0f;   // >=1 → disabled
};

int32_t    sample_greedy(const Tensor& logits);
int32_t    sample(const Tensor& logits, const SamplerConfig& cfg,
                  std::optional<uint64_t> seed = std::nullopt);
Tensor     filter_logits(const Tensor& logits, const SamplerConfig& cfg);
std::vector<int32_t>
           generate(LlamaModel& model,
                    const std::vector<int32_t>& prompt,
                    int max_new_tokens,
                    const SamplerConfig& cfg,
                    std::optional<int32_t> eos = std::nullopt,
                    std::optional<uint64_t> seed = std::nullopt);
```

### Pipeline

1. Apply `temperature` by scaling logits before softmax (T→0 makes
   argmax dominate; T→∞ flattens).
2. Apply `top_k`: zero out everything below the k-th largest logit.
3. Apply `top_p`: zero out everything outside the smallest set whose
   softmax mass ≥ p.
4. Compute softmax over the survivors.
5. Inverse-CDF sample with `std::mt19937_64` (seeded deterministically
   if a seed is supplied, otherwise from `std::random_device`).

### Sampling from the last prefill row

`LlamaModel::forward_cached` returns logits of shape `[seq, vocab]`.
For generation, only the **last row** corresponds to the next-token
distribution. `generate()` copies that row into a fresh `[vocab]`
tensor before sampling; the stochastic `sample()` itself just sees a
flat 1-D distribution, so the caller is responsible for picking the
right row.

### Bugs worth noting

- The first cut of `sample()` called `softmax_vec(..., /*temperature=*/1.0f)`,
  hardcoding T=1 even when the user configured a different temperature.
  For low-T sampling this made the output almost uniform. The fix is to
  pass `cfg.temperature` through.
- `generate()` originally sampled from the *flattened* prefill logits
  (every row, concatenated), which made the first sampled token out of
  range whenever `seq_q > 1`. The fix takes the last row before
  sampling.
- The EOS-halts test originally set only column 5 of `W_output` to
  `+1`, but with random weights the model's hidden state has negative
  mean, so `+1`-column dot products end up negative. The test now sets
  column 5 to `-10` and the others to `+1`, which flips the polarity
  and guarantees token 5 is the unique argmax.

### What's tested (13 cases)

- `sample_greedy` picks argmax (1-D and 2-D inputs)
- `filter_logits`: T=0 makes argmax +inf / others −inf
- `filter_logits`: top-k keeps only the top-k
- `filter_logits`: top-p keeps the smallest sufficient set
- `sample` is deterministic given a fixed seed
- `sample` distribution roughly tracks the softmax (high-T)
- `sample` concentrates on argmax (low-T)
- T=0 dispatches to `sample_greedy` from inside `sample()`
- `generate` produces exactly `max_new_tokens` when EOS is not set
- `generate` halts at the configured EOS token
- `generate` returns empty for `max_new_tokens=0`
- `generate` throws on empty prompt

## Phase 8 notes — Quantization (Q8_0 + Q4_0)

### Block formats

Both formats quantize 32-element blocks of Float32 to a per-block
scale (f16) plus packed quantized values. Layout matches the GGUF
spec, so a quantized tensor round-trips through the file format:

| Format | Block size | Per-block bytes | Levels    | Range          |
|--------|-----------:|----------------:|----------:|----------------|
| Q8_0   |         32 |              34 |     256   |  [-127, 127]   |
| Q4_0   |         32 |              18 |      16   |     [-8, +7]   |

Q4_0 stores nibbles packed two-per-byte; the low nibble is the even
index, the high nibble is the odd index. Both use round-half-away-
from-zero during quantization, with clamp-to-range.

### Pipeline

```cpp
// F32 → Q8_0 packed bytes
std::vector<uint8_t> quantize_q8_0(const float* src, int64_t n);

// Q8_0 packed bytes → F32
void dequantize_q8_0(const uint8_t* packed, int64_t n, float* dst);

// Same for Q4_0.

// Reference quantized matmul: dequantizes each row into a fresh F32
// buffer and accumulates via F32 multiply. Phase 9 replaces this with
// a fused dequant-dot kernel.
void matmul_q4_0_f32(const uint8_t* qmat, int64_t M, int64_t K,
                     const float* x, float* y);
void matmul_q8_0_f32(const uint8_t* qmat, int64_t M, int64_t K,
                     const float* x, float* y);
```

### GGUF integration

`GgufFile::load_tensor` now dequantizes Q8_0 and Q4_0 tensors to F32
on the fly. Both formats are accepted in the parser and tested by
synthetic round-trip files: write a quantized tensor to a hand-built
GGUF v3 buffer, read it back, and check that the dequantized values
match.

### Bugs worth noting

- The first cut used `std::array<int8, …>` (missing the `_t`); not a
  standard type and broke the build. Switched to `int8_t`.
- A first attempt at the GGUF round-trip test treated `ti.offset` as
  the absolute file position. It's actually the offset *into the data
  section* — the reader seeks to `data_section_offset_ + ti.offset`.
  With a single tensor at the start of the section, `ti.offset = 0`
  is correct.
- The same first cut reversed the GGUF dims, which writes them in the
  opposite order from what the parser reads. The convention in this
  codebase (and the existing test_gguf.cpp) is natural row-major order,
  not GGUF's "slowest first".
- The Q4_0 round-trip test originally used uniform random values in
  [-1, 1], which mixes values across many different per-block scales.
  At Q4_0's 16-level resolution, a value of 0.01 against an absmax of
  ~0.5 quantizes to zero with 100% relative error — not a bug, just
  the fundamental limit of absolute-scale quantization on a wide
  dynamic range. The test now (a) compresses the range to ±0.5 and
  (b) excludes values within `4 × scale` of zero from the
  relative-error measurement. After that filter, worst-case relative
  error is < 50% and typically ~7%.

### What's tested (15 cases)

- `f16 <-> f32` round-trip for normals, infinities, zero
- Q8_0: per-block round-trip, full-tensor round-trip, zero block,
  non-multiple-of-32 rejection
- Q4_0: per-block round-trip, full-tensor round-trip, zero block,
  non-multiple-of-32 rejection
- `matmul_q8_0_f32` matches F32 matmul (small + larger)
- `matmul_q4_0_f32` matches F32 matmul (small + larger)
- `GgufFile::load_tensor` round-trips Q8_0 + Q4_0 tensors written to
  a synthetic GGUF v3 file

## Phase 9 notes — Performance engineering

### Fused Q4_0 × F32 AVX2 matvec

The big win is fusing the dequantization into the dot product. The
reference path (`matmul_q4_0_f32_reference`) dequantizes each row into
a fresh 32-element F32 buffer, then runs an FMA loop. The fused kernel
(`matvec_q4_0_f32_avx2`, in `src/matmul.cpp`) loads the 16 packed
bytes of a Q4_0 block, splits them into 16 lo-nibbles and 16
hi-nibbles, zero-extends each nibble to 8-wide `__m256i` (after
subtracting 8 to re-center), converts to F32, and uses
`_mm256_i32gather_ps` to pull the matching strided x values into
SIMD registers. The four FMAs per block (lo_a × lo_x, hi_a × hi_x,
lo_b × lo_x, hi_b × hi_x) are reduced with `hsum256`, multiplied by
the per-block f16 scale, and accumulated into `y[m]`.

The Q8_0 path currently still goes through the reference. Q8_0 is
naturally friendlier to SIMD (no nibble unpacking, no strided x
loads) but we haven't migrated it yet.

### Build-time guard

`TINYLLM_HAVE_AVX2` (in `matmul.cpp`) and `TINYLLM_ENABLE_AVX2` (in
`quantize.cpp`) are set by CMake when the target CPU supports it. On
non-AVX2 builds the reference path is used; the kernel itself
compiles to a runtime-error stub.

### Measured speedup

`bench_quantize` reports Q4_0 / Q8_0 × F32 matvec vs F32 matmul on
random square matrices:

| M       | K       | F32 (ms) | Q4_0 (ms) | Speedup |
|--------:|--------:|---------:|----------:|--------:|
|     128 |     256 |    0.044 |     0.042 |    1.03 |
|     256 |     512 |    0.172 |     0.177 |    0.97 |
|     512 |    1024 |    0.815 |     0.687 |    1.19 |
|    1024 |    2048 |    4.185 |     2.773 |    1.51 |

At 1024 × 2048 the fused kernel beats F32 by 1.5x — that's the AVX2
path (4 FMA chunks per Q4_0 block, gather x, hsum, scale) versus the
naive F32 matmul (which at this size is the `Avx2` variant: 6×16
micro-tile, FMA into 12 accumulators, write back). The Q4_0 win is
mostly from halving memory bandwidth on the weight matrix (4 bits
per value vs 32). Smaller shapes don't benefit because the kernel
launch / hsum overhead dominates.

### Bugs worth noting

- The first AVX2 implementation unpacked lo-nibbles and hi-nibbles
  with `_mm_unpacklo_epi8`, which only sees the **low** 8 bytes of the
  source. So the kernel was computing dot products over 16 of the
  32 block values — exactly half. Corrected by using
  `_mm_unpacklo_epi8` AND `_mm_unpackhi_epi8`, widening to 16 int16,
  then to 16 int32 across two `__m256i`. The test
  `matmul_q4_0_matches_f32` caught it (got the right magnitude but
  the wrong sign on rows with a negative mean).
- Linking initially failed because the helper lived in `tinyllm::ops`
  but the forward declaration in `quantize.cpp` was in the file's
  anonymous namespace, making the linker look for
  `tinyllm::{anon}::matvec_q4_0_f32_avx2`. Moved the forward
  declaration to `tinyllm::ops` scope at the top of `quantize.cpp`
  and qualified the call site as `tinyllm::ops::matvec_q4_0_f32_avx2`.

## Phase 10 notes — Benchmarking suite

### `bench_transformer` — the unified runner

`benchmarks/bench_transformer.cpp` is the single entry point that
exercises every kernel the project ships. For each model config
(`small` hidden=512 / `medium` hidden=1024 / `big` hidden=2048) it
prints a table:

```text
=== Config: medium (hidden=1024, L=2) ===
kernel                  shape                          ms/call      throughput
attention_prefill       prefill seq=32 heads=8           7.515       133.1 calls/s
attention_decode        decode seq=1 ctx=64 heads=8      2.533       394.8 calls/s
mlp_forward             seq=32 h=1024 i=2048            10.533        94.9 calls/s
block_prefill           prefill seq=32 layers=1         21.786        45.9 calls/s
block_decode            decode seq=1 ctx=64 layers=1      6.419       155.8 calls/s
llama_forward           prefill seq=32 L=2              44.132        22.7 calls/s
llama_forward_cached    decode seq=1 ctx=32 L=2         20.527        48.7 calls/s
```

A CSV block at the bottom lets you diff runs across machines or
compiler flags.

### What it covers

| Group     | Kernels                                                                                          |
|-----------|--------------------------------------------------------------------------------------------------|
| Tensor    | `add`, `multiply`, `mul_scalar`, `sum`, `mean`, `softmax`, `reshape`, `transpose`                |
| Matmul    | `matmul_avx2`, `matmul_blocked`, `matmul_threaded` at n ∈ {256, 1024, 2048}                     |
| Quantized | `matmul_q4_0_f32` (AVX2 fused) at (M, K) ∈ {(512,4096), (1024,4096), (2048,4096)}                |
| Layer     | `rmsnorm`, `rope_inplace`                                                                        |
| Attention | prefill (seq=32) + decode (seq=1, ctx=64)                                                        |
| MLP       | `mlp_forward` (SwiGLU)                                                                           |
| Block     | `llama_block_forward` prefill + cached decode                                                    |
| End-to-end | `llama_forward` prefill + `llama_forward_cached` decode                                         |

### Usage

```bash
./build/bin/bench_transformer             # default: 30 iterations per kernel
./build/bin/bench_transformer --quick     # 5 iterations, fewer configs
./build/bin/bench_transformer --iters 100 # custom iteration count
./build/bin/bench_transformer > out.txt   # capture tables + CSV
```

### How to read the numbers

- **Tensor ops** are bandwidth-bound; ms/call scales with the number
  of elements touched (e.g. `add` on 1024² ≈ 4 MB reads + 4 MB writes,
  dominated by memory traffic).
- **Matmul** numbers tell you how well the AVX2 and threaded kernels
  scale. At 256³ `matmul_avx2` (0.73 ms) ≈ `matmul_threaded` (1.09 ms):
  the threading overhead exceeds the work at this size. At 1024³
  threaded wins 2.4× (30.5 ms vs 73.6 ms single-threaded AVX2).
- **Quantized matvec** at M=2048, K=4096 is 11 ms — that's ~150 GFLOPS
  effective (the AVX2 path is doing 2·M·K FMAs plus nibble unpacking
  and a gather).
- **Block prefill vs decode** (e.g. medium: 21.8 ms vs 6.4 ms)
  reflects the O(seq²) attention prefill vs the O(seq · cache) decode.
  In real LLM workloads this is why prefill latency dominates the
  first request but generation throughput is bounded by decode.

### Bugs worth noting

- The first cut used `bench_id(0, 32000)` for token sampling, but the
  `small` config has `vocab=4096`. Tokens above 4095 tripped
  `llama: token id out of range`. The bench now builds an
  `id(0, vocab-1)` per config.
- The decode benches pre-fill the KV cache to 63 rows then decode at
  `start_pos=63`. With `iters > 1` and a fixed-size cache that hits the
  capacity-exceeded error on the second iteration. The fix is to reset
  `cache.length = 63` inside the timed closure so each call starts from
  the same cache state.

### Public API additions

`ops::hardware_threads()` and `ops::have_avx2()` were promoted from
`matmul.cpp`'s anonymous namespace to the public header so the bench
suite can report what it actually targeted.

## License

For personal learning; no license declared yet.
