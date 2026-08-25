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
|  11   | GGUF → LlamaModel loader + driver    |   ✅   |
|  12   | K-quant dequant (Q4_K / Q5_K / Q6_K) |   ✅   |
|  13   | AVX-512 fused Q4_0 matvec             |   ✅   |
|  14   | Fused Q4_K / Q6_K × F32 matvec (AVX2) |   ✅   |

## Build

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Requirements: CMake ≥ 3.20, a C++20 compiler (GCC 11+, Clang 14+), Ninja.

## Test & benchmark

```bash
./build/bin/run_all_tests    # 136 unit tests (23 tensor, 7 matmul, 13 tokenizer, 14 gguf,
                             # 7 rmsnorm, 8 rope, 5 attention, 3 mlp, 5 model, 7 kv_cache,
                             # 13 sampler, 29 quantize, 6 llama_loader)
./build/bin/bench_tensor     # naive matmul baseline numbers
./build/bin/bench_matmul     # naive / blocked / AVX2 / threaded comparison
./build/bin/bench_quantize   # Q4_0 / Q8_0 / Q4_K / Q6_K × F32 matvec vs F32 matmul
./build/bin/bench_transformer # unified Phase 10 suite (all kernels, end-to-end)
./build/bin/bench_tokenizer  # BPE encode throughput
./build/bin/phase1_demo     # 2-layer MLP smoke test
./build/bin/tinyllm         # CLI smoke test
./build/bin/generate_demo   # end-to-end inference with random-init Llama
                            # (greedy + top-k + top-p + sampling + Q4_0 demo)
./build/bin/gguf_driver     # load a real Llama-family GGUF and run generate
                            # (token-IDs in, token-IDs out; no tokenizer shim)
```

## Load a real model

`examples/gguf_driver.cpp` is the closest thing we ship to "actually run a
real LLM". It loads any Llama-family GGUF v3 file (TinyLlama, Qwen2,
Llama-2, etc.), materializes the weights as F32 (Q4_0/Q8_0/F16 are
dequantized on the fly by `GgufFile::load_tensor`), and runs
`generate(...)` over a prompt.

```bash
# Pick the prompt token IDs with whatever tokenizer produced the GGUF,
# then feed them in:
./build/bin/gguf_driver \
    --gguf tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
    --prompt 1,2,3,4,5 \
    --max-new 32 \
    --temperature 0.8 \
    --top-p 0.95 \
    --seed 42
```

The CLI prints token IDs (decimal and hex) and a `tok/s` number. There
is no built-in tokenizer shim — convert text → ids with the tokenizer
that produced the model, and ids → text with the same. This is the
deliberate boundary: tokenization is a per-model problem and the engine
itself shouldn't care.

If the estimated F32 weight memory exceeds 4 GiB, the loader prints a
soft warning before proceeding — handy when you accidentally point it
at a 7B Q4_0 and were about to materialize ~28 GiB of float32.

## Layout

```text
tinyllm/
├── CMakeLists.txt
├── include/tinyllm/
│   ├── memory.hpp       # ref-counted aligned storage
│   ├── tensor.hpp       # Tensor + ops + broadcasting + views
│   ├── matmul.hpp       # matmul variants + MatmulVariant enum
│   ├── tokenizer.hpp    # BPE tokenizer
│   ├── gguf.hpp         # GGUF file parser
│   ├── quantize.hpp     # Q4_0, Q8_0, Q4_K, Q5_K, Q6_K block formats
│   └── llama_loader.hpp # GGUF → LlamaModelWeights
├── src/
│   ├── memory.cpp
│   ├── tensor.cpp
│   ├── matmul.cpp       # naive / blocked / AVX2 / threaded
│   ├── tokenizer.cpp
│   ├── gguf.cpp
│   ├── quantize.cpp     # dequant kernels for Q4_0/Q8_0 and the
│   │                    # K-quants (Q4_K / Q5_K / Q6_K)
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

## Phase 11 notes — GGUF → LlamaModel loader

### What it does

`include/tinyllm/llama_loader.hpp` + `src/llama_loader.cpp` close the
last functional gap: the GGUF parser could *read* a Llama checkpoint,
but nothing turned a parsed GGUF into a runnable `LlamaModel`. The
loader does exactly that — read the metadata KVs, walk `tensor_infos()`
by name, dequantize each named weight to F32 (the existing
`GgufFile::load_tensor` already does the Q4_0/Q8_0/F16 → F32 work),
and populate a `LlamaModelWeights` ready for `make_model`.

### Public surface

```cpp
struct LlamaLoadResult { LlamaConfig cfg; LlamaModelWeights weights; };
std::optional<LlamaLoadResult> load_llama_from_gguf(
    const std::string& path, std::ostream& diag);
std::size_t estimate_llama_f32_bytes(const GgufFile& gf);
```

Errors are `std::optional`-shaped: `nullopt` means the file could not
be opened, a required KV was missing or had the wrong type, or a
required tensor was missing. The diagnostic stream receives one line
per issue so the user can fix their GGUF.

### Naming conventions

The loader recognizes the standard llama.cpp / ggml convention used by
basically every Llama-family quantizer:

```text
token_embd.weight         → W_embed[vocab, hidden]
output.weight             → W_output[hidden, vocab]   (skipped if absent)
output_norm.weight        → final_norm[hidden]
blk.{i}.attn_norm.weight  → blocks[i].attn_norm
blk.{i}.attn_q.weight     → blocks[i].attn.Wq
blk.{i}.attn_k.weight     → blocks[i].attn.Wk
blk.{i}.attn_v.weight     → blocks[i].attn.Wv
blk.{i}.attn_output.weight→ blocks[i].attn.Wo
blk.{i}.ffn_norm.weight   → blocks[i].mlp_norm
blk.{i}.ffn_gate.weight   → blocks[i].mlp.W_gate
blk.{i}.ffn_up.weight     → blocks[i].mlp.W_up
blk.{i}.ffn_down.weight   → blocks[i].mlp.W_down
```

`LlamaConfig` is built from `llama.vocab_size`, `llama.embedding_length`,
`llama.feed_forward_length`, `llama.attention.head_count`,
`llama.attention.head_count_kv`, `llama.block_count`, `llama.context_length`,
`llama.attention.layer_norm_rms_epsilon` (optional),
`llama.rope.freq_base` (optional). `head_dim` is derived as
`hidden / n_heads`.

### Tied embeddings

Real Llama 2/3 checkpoints don't include `output.weight` — they tie it
to `W_embed` transposed. When `output.weight` is absent the loader
emits a diagnostic and copies a transposed view of `W_embed` into
`W_output`. Llama-2-7B-instruct ships this way; TinyLlama does not.

### Memory soft-warning

Every named tensor's on-disk byte size is summed and multiplied by 4
(the worst-case F32 inflation ratio: Q4_0 is ~7× but with the
embeddings/head counted at full size the *average* multiplier is
smaller — we use 4× as a rough upper bound). If that exceeds
`kLlamaLoadSoftWarnBytes` (4 GiB by default), the loader prints a
warning to the diagnostic stream and **proceeds anyway**. This is the
deliberate hard-fail boundary: loading a Q4_0 7B will materialize
~28 GiB of float32, and the user should see that before it happens.

### `gguf_driver`

`examples/gguf_driver.cpp` wires the loader to `generate(...)` and
prints the output as token IDs:

```bash
./build/bin/gguf_driver --gguf model.gguf --prompt 1,2,3 --max-new 16
```

The driver prints the loaded `LlamaConfig`, the estimated F32
footprint, the prompt and generated token sequences (decimal and hex),
and a `tok/s` number. Greedy runs are checked for determinism: the
generator is invoked twice with the same seed and the outputs are
compared. Token-IDs-only is a deliberate scope choice — a tokenizer
shim is per-model-family and out of scope for the engine itself.

### Tests

`tests/test_llama_loader.cpp` adds 6 tests:

- `loader_populates_config` — every LlamaConfig field round-trips.
- `loader_finds_every_named_tensor` — every required slot is filled
  with the right shape.
- `loader_fails_cleanly_on_missing_tensor` — when the file omits
  required tensors, the loader returns nullopt and names what's
  missing.
- `loader_round_trips_a_real_shape_model` — the loaded weights, fed
  through `make_model` + a forward pass, produce logits byte-equal
  (within 1e-5) to a hand-built reference model built from the same
  tensors.
- `loader_estimates_f32_memory` — `estimate_llama_f32_bytes` agrees
  with the expected sum and is well below the soft-warn threshold for
  the synthetic file.
- `loader_handles_tied_embeddings` — a file with no `output.weight`
  yields a populated `W_output` (transposed from `W_embed`) and the
  diagnostic mentions "tying W_output".

The synthetic file fixture is also dumped to
`/tmp/tinyllm_test_loader.gguf` by the last test, so the driver can be
smoke-tested without a real Llama checkpoint on hand.

## Phase 12 notes — K-quant dequant (Q4_K / Q5_K / Q6_K)

### Why

Phase 11 shipped the GGUF → LlamaModel loader, but real-world Llama
checkpoints (TinyLlama, Llama-2, Qwen2, Mistral, …) don't actually ship
as `Q4_0` — they ship as **`Q4_K_M`**, `Q5_K_M`, or `Q6_K`. `load_tensor`
threw "unsupported dtype" for any of those, which meant the
`gguf_driver` could only load the synthetic test fixture. Phase 12 adds
the missing dequant paths so `gguf_driver` can load any real
Llama-family GGUF.

### What got added

`include/tinyllm/gguf.hpp` extended the `GgufTensorType` enum with
`Q2_K = 10, Q3_K = 11, Q4_K = 12, Q5_K = 13, Q6_K = 14`. We only
implement the read paths for Q4_K / Q5_K / Q6_K — Q2_K and Q3_K remain
"unsupported dtype" because no mainstream quantizer produces them any
more (and they'd need fused dequant + matvec to be useful).

`src/quantize.cpp` got three new dequant kernels — verbatim ports of
`llama.cpp`'s `dequantize_row_q4_K`, `dequantize_row_q5_K`,
`dequantize_row_q6_K`. The reference C (Apache-2.0) lives in
`ggml-quants.c` and uses a clever nibble-share scheme for the 6-bit
`(scale, min)` table of Q4_K/Q5_K that isn't documented in the GGUF
spec; following the reference is the only safe way to read these
formats.

```cpp
// All three are dequantize only — the F32 loader dequantizes into the
// weight tensor; the existing F32 matmul is reused. Reference
// `matmul_q4_K_f32` and `matmul_q6_K_f32` are exposed for parity with
// the Q4_0/Q8_0 paths; no fused K-quant kernel yet.
void dequantize_q4_K(const uint8_t* packed, int64_t n, float* dst);
void dequantize_q5_K(const uint8_t* packed, int64_t n, float* dst);
void dequantize_q6_K(const uint8_t* packed, int64_t n, float* dst);
void matmul_q4_K_f32(const uint8_t* qmat, int64_t M, int64_t K,
                     const float* x, float* y);
void matmul_q6_K_f32(const uint8_t* qmat, int64_t M, int64_t K,
                     const float* x, float* y);
```

`src/gguf.cpp::GgufFile::load_tensor` got three new switch cases for
`Q4_K`, `Q5_K`, `Q6_K` that read the packed bytes from disk and call
into the dequant kernels — same pattern as the existing Q4_0/Q8_0
cases. `src/llama_loader.cpp::tensor_byte_size` was extended with the
matching block sizes so the >4 GiB soft-warning remains accurate.

`GgufTensorType::Q2_K` and `Q3_K` are recognized by name (so loading a
Q2_K file gives a clear "unsupported dtype" message rather than "bad
tensor type id") but the dequant is not implemented — they'd be a
follow-up phase if we ever need them.

### Format recap

```text
QK_K = 256  (super-block size, except Q6_K which is 16 sub-blocks of 16)

Q4_K super-block (144 bytes):
    d : f16  |  dmin : f16  |  12-byte packed 6-bit (sc, m)  |  128 qs
  dequant:  y[i] = d * sc[i/32] * qs_nibble - dmin * m[i/32]

Q5_K super-block (176 bytes):
    d : f16  |  dmin : f16  |  12-byte packed 6-bit (sc, m)
                                          |  128 qs  |  32 qh (1 bit/elt)
  dequant:  y[i] = d * sc[i/32] * (qs_nibble + (qh[i] ? 16 : 0)) -
                    dmin * m[i/32]

Q6_K super-block (210 bytes):
    128 ql  |  64 qh  |  16 scales int8  |  2 d : f16
  dequant:  y[i] = d * sc[i/16] * ((ql | (qh << 4)) - 32)
```

### Tests

`tests/test_quantize.cpp` adds 11 tests:

- `q4_K_dequant_matches_formula`, `q5_K_dequant_matches_formula`,
  `q6_K_dequant_matches_formula` — hand-construct a super-block with
  known `d`, `dmin`, scales, and `qs/qh` values, dequantize, and verify
  the output matches the formula element by element.
- `q4_K_zero_block_dequantizes_to_zero` and the Q5_K / Q6_K equivalents
  — all-zero packed input dequantizes to all-zero F32 output.
- `gguf_loads_q4_K_tensor_all_zero` and the Q5_K / Q6_K equivalents —
  write a synthetic GGUF v3 file with one K-quant tensor, load it via
  `GgufFile::load_tensor`, and verify the result is an all-zero F32
  tensor of the right shape.

The reader is one straight-line port from `llama.cpp`'s reference
implementation — the tests are the safety net that catches any
arithmetic typo. They all pass.

### What's still missing

- Q2_K / Q3_K support. Nobody ships them in 2025+ (Q4_K dominates), so
  these remain "unsupported dtype".
- **A fused** K-quant matvec kernel — *addressed in Phase 14*.
  Phase 12 leaves the K-quant matmul on the dequant-bounce path; the
  fused Q4_K / Q6_K kernel ships in the next phase.

## Phase 13 notes — AVX-512 fused Q4_0 matvec

### Why

Phase 9 shipped the AVX2 fused Q4_0 kernel (`matvec_q4_0_f32_avx2`),
which is the hot path for every Q4_0 weight × F32 activation in the
model. Phase 13 ports that kernel to AVX-512F, the natural follow-up:
on x86 servers and modern desktop CPUs (Skylake-X / Zen 4 / Sapphire
Rapids and later), 512-bit registers and the wider FMA / gather
pipeline roughly **1.4×–1.9× faster** for this kernel — see numbers
below.

### What got added

- `CMakeLists.txt` — a second `check_cxx_compiler_flag` for
  `-mavx512f -mavx512vl -mavx512bw -mavx512vbmi2`. When the compiler
  accepts the flag, we add it to `tinyllm_core` (now `PUBLIC` so the
  `bench_*` targets get the same defines and can call into the kernel
  directly) and define `TINYLLM_ENABLE_AVX512=1`.
- `src/matmul.cpp`:
  - New `TINYLLM_HAVE_AVX512` macro, set iff
    `defined(__AVX512F__) && defined(__AVX512VL__) && defined(__AVX512BW__)`
    (we don't actually need VL/BW for the kernel itself, but we gate
    together to avoid skew).
  - New `matvec_q4_0_f32_avx512(...)` kernel in `tinyllm::ops`. Same
    algorithm as the AVX2 path (nibble unpack → subtract 8 → multiply
    by strided x → hsum), but each 32-element Q4_0 block fits in
    **two** `__m512` vectors instead of four `__m256`s, and the hsum
    collapses to one `_mm512_reduce_add_ps`. The dispatch
    (`pick_best`) and the public `CpuFeatures` struct gained an
    `avx512` field.
  - New `have_avx512()` accessor on `tinyllm::ops` for the
    bench/feature-detection tests.
  - New direct kernel entry points (`matvec_q4_0_f32_avx2` and
    `matvec_q4_0_f32_avx512`) declared in `matmul.hpp` so the benchmark
    suite can call them head-to-head.
- `src/quantize.cpp`:
  - `matmul_q4_0_f32` dispatch chain is now
    `AVX-512 → AVX2 → reference`. The static `TINYLLM_ENABLE_AVX512`
    flag chooses; a future runtime CPUID probe would let us ship a
    single binary that picks per-process.
- `benchmarks/bench_quantize.cpp`:
  - Header line prints `AVX2=1 AVX-512=1 threads=N` so the user
    immediately sees what the build actually targets.
  - For each shape, the bench now also reports `Q4_0_avx2` and
    `Q4_0_avx512` separately, plus a one-line
    `# speedup avx512 vs avx2: 1.6x` summary.
- `benchmarks/bench_transformer.cpp` — CPU-feature line now prints
  AVX-512 too.
- `tests/test_quantize.cpp` — new parity test
  `matvec_q4_0_avx2_vs_avx512` runs both kernels on the same random
  input and asserts agreement within the same tolerance the F32 path
  uses (1e-2 × K).

### Numbers

`bench_quantize` on a 4-thread Skylake-class CPU with both AVX2 and
AVX-512 enabled:

```text
format,M,K,f32_ms,qmat_ms,speedup
Q4_0,128,256,0.094,0.033,2.88x
Q4_0_avx2,128,256,,0.046,
Q4_0_avx512,128,256,,0.029,
# speedup avx512 vs avx2: 1.59x at (128,256)
Q4_0,256,512,0.266,0.111,2.40x
Q4_0_avx2,256,512,,0.176,
Q4_0_avx512,256,512,,0.120,
# speedup avx512 vs avx2: 1.47x at (256,512)
Q4_0,512,1024,0.901,0.442,2.04x
Q4_0_avx2,512,1024,,0.855,
Q4_0_avx512,512,1024,,0.457,
# speedup avx512 vs avx2: 1.87x at (512,1024)
Q4_0,1024,2048,4.066,1.853,2.19x
Q4_0_avx2,1024,2048,,2.960,
Q4_0_avx512,1024,2048,,2.165,
# speedup avx512 vs avx2: 1.37x at (1024,2048)
```

The shape dependence is the expected gather-throughput effect: AVX-512
helps the most on smaller K (where per-block setup dominates) and the
least on larger K (where the two gathers dominate and they're roughly
the same speed on both ISAs).

### What's still missing

- **Runtime CPUID dispatch.** Today the choice is compile-time; we
  could ship a single binary that probes and picks. Trivial follow-up
  if needed.
- **AVX-512 VNNI** for Q8_0. `vpdpwssd` would let us skip the per-lane
  int8→int16 promotion and FMA int8×int8 directly into int32
  accumulators. ~2× for Q8_0 if we ever care.
- **AVX-512 BFloat16** for F32 matmul itself. Would need the F32
  weights converted at load time; not worth it until there's a
  measured F32 matmul bottleneck in `bench_transformer`.
- **Q5_K fused matvec.** Q5_K is rarely used in practice (Q4_K + Q6_K
  dominate), so it's dequant-only here. The same AVX2 pattern would
  apply with the qh bit added to each nibble.

## Phase 14 notes — Fused Q4_K / Q6_K × F32 matvec (AVX2)

### Why

Most real Llama-family checkpoints are quantized with the **K-quant**
formats (`Q4_K`, `Q5_K`, `Q6_K`), not `Q4_0`. TinyLlama-1.1B Q4_0
exists but the majority of the model zoo — Qwen2, Llama-3, Mistral,
Gemma — ships in Q4_K or Q6_K. Phase 12 added the dequant path so we
could *load* these weights (via `load_tensor`); Phase 14 closes the
loop with a fused on-the-fly dequant+FMA kernel so we don't pay the
M×K dequant-bounce for every token.

### What got added

- `include/tinyllm/matmul.hpp` — declared
  `matvec_q4_K_f32_avx2` and `matvec_q6_K_f32_avx2` so the bench suite
  can call them directly.
- `src/matmul.cpp` — two new kernels in the anonymous namespace:
  - **`matvec_q4_K_f32_avx2`** — Q4_K super-block = `[d:f16][dmin:f16][scales:12][qs:128]`.
    Eight sub-blocks of 32 elements each. The dequant for one
    sub-block is `d * sc * (qs - 0) + (-dmin) * m`. Because Q4_K packs
    lo+hi nibbles in stride (lo→x[0..31], hi→x[32..63]) rather than
    Q4_0's stride-2, the FMA pattern is "load 8 contiguous x's, FMA
    into the right sub-block accum" — the Q4_0 kernel's strided x
    gather trick wouldn't work here.
  - **`matvec_q6_K_f32_avx2`** — Q6_K super-block =
    `[ql:128][qh:64][scales:16 (int8)][d:f16]`. 16 sub-blocks of 16.
    Each output value is `d * sc * (ql + (qh&3)<<4 - 32)`. The kernel
    walks the two 16-byte `ql` halves per "half" (l=0..15 vs l=16..31)
    and the corresponding 16-byte `qh` half, processes 4 sub-blocks
    × 2 halves × 16 elements, and accumulates into the row sum.
- `src/quantize.cpp`:
  - **`matmul_q4_K_f32`** / **`matmul_q6_K_f32`** dispatch is now
    `AVX2 → reference` (vs the previous always-reference).
  - **Bug fix in `dequantize_q4_K` / `dequantize_q5_K` / `dequantize_q6_K`**:
    all three wrote every super-block to `dst[0..255]` instead of
    `dst[i*kQK_K + ...]`. The reference path was used both by the
    "no-AVX2" matmul and by the tests' reference comparison, so the
    bug silently cancelled out — both paths agreed on the wrong
    answer. The fused kernel didn't have that luxury, so we fixed it
    here.
- `tests/test_quantize.cpp` — two new parity tests
  `matmul_q4_K_matches_f32` and `matmul_q6_K_matches_f32`. Each
  constructs a random-but-valid Q4_K or Q6_K buffer, runs the fused
  kernel, then runs the dequant-then-FMA reference on the same
  buffer, and asserts agreement.
- `benchmarks/bench_quantize.cpp` and `benchmarks/bench_transformer.cpp` —
  new rows for Q4_K and Q6_K at the same shapes as Q4_0, so we can
  put numbers on the fused-vs-dequant speedup.

### Numbers

`bench_quantize` (release build, AVX2 + AVX-512, single-threaded):

```text
Q4_0,512,1024,0.827,0.467,1.77x
Q4_K,512,1024,0.827,0.110,7.51x
Q6_K,512,1024,0.827,0.118,7.00x
Q4_0,1024,2048,3.116,1.743,1.79x
Q4_K,1024,2048,3.116,0.451,6.91x
Q6_K,1024,2048,3.116,0.472,6.61x
```

The numbers are vs F32 matmul; the more interesting comparison is
fused K-quant vs the dequant-bounce reference, which gets us back
to a single Q4_0-style number while still paying only ~30% of the
F32 cost.

The fused K-quant throughput (`bench_transformer`'s quantized-matvec
table, 4096-wide K, single-threaded):

```text
matmul_q4_0_f32  512x4096   1.888 ms
matmul_q4_K_f32  512x4096   0.477 ms
matmul_q6_K_f32  512x4096   0.491 ms
matmul_q4_0_f32 1024x4096   3.488 ms
matmul_q4_K_f32 1024x4096   0.905 ms
matmul_q6_K_f32 1024x4096   0.959 ms
```

Q4_K and Q6_K are ~2× the throughput of Q4_0 at this width because
the FMA pattern hits 8 of 8 lanes every cycle, vs Q4_0's
stride-2 gather which only hits ~4 of 8.

### Bugs worth noting

- **`_mm_srli_epi16` shifts 16-bit WORDS, not bytes.** Earlier draft
  of the Q6_K kernel tried to shift a packed-byte `__m128i` by 2 with
  `_mm_srli_epi16`; the right primitive is to widen via
  `_mm256_cvtepu8_epi32` then shift int32s with
  `_mm256_srli_epi32`.
- **`_mm256_srli_si256` shifts each 128-bit lane by *bytes*, not
  across the whole 256-bit register.** To move 8 bytes from the low
  lane to the high lane, use `_mm_unpackhi_epi64` to get the high 8
  bytes as a fresh `__m128i`, then `_mm256_cvtepu8_epi32` on that.

### What's still missing

- **AVX-512 fused K-quants.** Same AVX-512F port as Phase 13 for Q4_0
  would give another ~1.5× on Skylake-X / Zen 4 / Sapphire Rapids.
  Most laptop/desktop CPUs don't have AVX-512 so AVX2 is the
  more-portable target.
- **Runtime CPUID dispatch** for the AVX2 path. Today
  `TINYLLM_ENABLE_AVX2` is a compile-time choice.
- **A bench row for end-to-end Q4_K inference** (loaded GGUf → 100
  tokens). The existing `gguf_driver` runs the full path, just not in
  the bench harness.

## License

For personal learning; no license declared yet.
