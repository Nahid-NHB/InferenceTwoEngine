// include/tinyllm/llama_loader.hpp
// -----------------------------------------------------------------------------
// GGUF -> LlamaModelWeights loader for Llama-family checkpoints.
//
// Reads a GGUF v3 file, extracts the LlamaConfig from the metadata KV pairs
// (key naming follows llama.cpp / ggml: `llama.vocab_size`,
// `llama.embedding_length`, ...), and copies every named weight tensor
// (`token_embd.weight`, `blk.{i}.attn_q.weight`, ...) into a populated
// `LlamaModelWeights` ready to be wrapped by `make_model`.
//
// `load_tensor` always dequantizes Q4_0 / Q8_0 to F32 internally, so the
// loader works against any GGUF regardless of weight dtype, at the cost of
// materializing the full F32 weight set in RAM.
//
// Per-tensor F32 inflation is estimated by summing the on-disk byte sizes
// of every tensor the loader needs (using `byte_size = numel *
// gguf_tensor_type_size(...)` with a Q4_0/Q8_0 fallback). If the estimated
// F32 size exceeds a soft threshold (4 GB by default), a warning is
// emitted to the diagnostic stream but loading still proceeds.
//
// Errors are reported via std::optional: nullopt means the file could not
// be opened, a required KV was missing or had the wrong type, or a
// required tensor was missing. The diagnostic stream receives one line per
// missing item so the user can fix their GGUF.
// -----------------------------------------------------------------------------
#pragma once

#include "tinyllm/model.hpp"

#include <iosfwd>
#include <optional>
#include <string>

namespace tinyllm {

struct LlamaLoadResult {
    LlamaConfig       cfg;
    LlamaModelWeights weights;
};

// Soft-warning threshold (in bytes) for estimated F32 weight memory.
// 4 GiB is enough headroom for a Q4_0 1.1B model after dequant; loading
// a Q4_0 7B will trip the warn and let the user confirm they really want
// the ~28 GiB F32 footprint.
constexpr std::size_t kLlamaLoadSoftWarnBytes = 4ULL * 1024 * 1024 * 1024;

// Load a Llama-family model from a GGUF file. On success returns the
// config + populated weights; on any structural problem (missing tensor,
// missing KV, wrong type) returns std::nullopt and writes a human-readable
// diagnostic to `diag` (one line per issue).
std::optional<LlamaLoadResult> load_llama_from_gguf(
    const std::string& path,
    std::ostream&      diag);

// Estimate the F32-dequantized memory footprint (bytes) for the tensors
// named by the Llama convention in `gf`. Used to drive the soft warning;
// exposed so the driver / tests can also report it.
std::size_t estimate_llama_f32_bytes(const class GgufFile& gf);

}  // namespace tinyllm