// src/llama_loader.cpp
// -----------------------------------------------------------------------------
// Implementation of the Llama GGUF loader.
//
// Naming conventions follow llama.cpp / ggml, the same convention that
// basically every Llama-family quantizer emits:
//   token_embd.weight         -> W_embed [vocab, hidden]
//   output.weight             -> W_output [hidden, vocab]   (skipped if absent)
//   output_norm.weight        -> final_norm [hidden]
//   blk.{i}.attn_norm.weight  -> blocks[i].attn_norm
//   blk.{i}.attn_q.weight     -> blocks[i].attn.Wq
//   blk.{i}.attn_k.weight     -> blocks[i].attn.Wk
//   blk.{i}.attn_v.weight     -> blocks[i].attn.Wv
//   blk.{i}.attn_output.weight-> blocks[i].attn.Wo
//   blk.{i}.ffn_norm.weight   -> blocks[i].mlp_norm
//   blk.{i}.ffn_gate.weight   -> blocks[i].mlp.W_gate
//   blk.{i}.ffn_up.weight     -> blocks[i].mlp.W_up
//   blk.{i}.ffn_down.weight   -> blocks[i].mlp.W_down
//
// LlamaConfig metadata keys (also llama.cpp convention):
//   llama.vocab_size                          : uint32
//   llama.embedding_length                    : uint32
//   llama.feed_forward_length                 : uint32
//   llama.attention.head_count                : uint32
//   llama.attention.head_count_kv             : uint32
//   llama.block_count                         : uint32
//   llama.context_length                      : uint32
//   llama.attention.layer_norm_rms_epsilon    : float32 (optional)
//   llama.rope.freq_base                      : float32 (optional)
//
// `head_dim` is derived as embedding_length / head_count.
// `max_seq_len` defaults to `context_length` and may be overridden.
// -----------------------------------------------------------------------------
#include "tinyllm/llama_loader.hpp"

#include "tinyllm/gguf.hpp"
#include "tinyllm/quantize.hpp"

#include <cstdint>
#include <cstdio>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tinyllm {

namespace {

// Compute the on-disk byte size for a single tensor (numel elements of
// the given GGUF dtype). F32/F16 use the per-element size; Q4_0/Q8_0 use
// the block-quantized sizes. Anything else returns 0 (treated as
// unsupported by load_tensor).
std::size_t tensor_byte_size(int64_t numel, GgufTensorType dt) {
    switch (dt) {
        case GgufTensorType::F32: return static_cast<std::size_t>(numel) * 4;
        case GgufTensorType::F16: return static_cast<std::size_t>(numel) * 2;
        case GgufTensorType::Q4_0:
            if (numel % kQ4_0BlockSize != 0) return 0;
            return static_cast<std::size_t>(numel / kQ4_0BlockSize) * kQ4_0BlockBytes;
        case GgufTensorType::Q8_0:
            if (numel % kQ8_0BlockSize != 0) return 0;
            return static_cast<std::size_t>(numel / kQ8_0BlockSize) * kQ8_0BlockBytes;
        default: return 0;
    }
}

// F32-dequantized byte size = numel * 4. Pure helper for clarity at the
// call sites that care about the inflated footprint.
constexpr std::size_t f32_byte_size(int64_t numel) {
    return static_cast<std::size_t>(numel) * 4;
}

// Helper to read a typed KV; logs to diag and returns nullopt if absent
// or wrong type.
template <typename Ret, typename Accessor>
std::optional<Ret> read_kv(const GgufFile& gf, const std::string& key,
                           Accessor acc, std::ostream& diag,
                           const char* type_name) {
    const GgufValue* v = gf.get_kv(key);
    if (v == nullptr) {
        diag << "  - missing metadata KV: " << key << "\n";
        return std::nullopt;
    }
    auto x = acc(*v);
    if (!x.has_value()) {
        diag << "  - wrong type for KV: " << key << " (expected " << type_name
             << ", got " << gguf_value_type_name(v->type()) << ")\n";
        return std::nullopt;
    }
    return x;
}

// Move a Tensor's storage into a freshly-constructed Tensor by aliasing
// the existing storage. Used to avoid a second copy when the GGUF loader
// already dequantized into a contiguous F32 buffer. Since `Tensor` does
// not expose a "steal storage" constructor, we instead `contiguous()` the
// source (which is a no-op when the source is already contiguous F32).
Tensor take_loaded(Tensor src) {
    // All tensors returned by GgufFile::load_tensor are contiguous F32.
    // `.contiguous()` returns a view-equivalent copy only if the input
    // wasn't contiguous; here it returns the same data. This gives us a
    // self-contained Tensor without aliasing the GGUF reader's buffer.
    return src.contiguous();
}

// Apply a Tensor into the right slot of LlamaModelWeights, asserting the
// shape matches what we expect from cfg.
void check_shape(const Tensor& t, const std::vector<int64_t>& expected,
                 const std::string& name, std::ostream& diag) {
    if (t.shape() != expected) {
        diag << "  - shape mismatch for " << name << ": got [";
        for (std::size_t i = 0; i < t.shape().size(); ++i) {
            if (i) diag << ", ";
            diag << t.shape()[i];
        }
        diag << "], expected [";
        for (std::size_t i = 0; i < expected.size(); ++i) {
            if (i) diag << ", ";
            diag << expected[i];
        }
        diag << "]\n";
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Public surface
// ---------------------------------------------------------------------------

std::size_t estimate_llama_f32_bytes(const GgufFile& gf) {
    std::size_t total = 0;
    for (const auto& ti : gf.tensor_infos()) {
        int64_t n = 1;
        for (uint64_t d : ti.dims) n *= static_cast<int64_t>(d);
        std::size_t bytes = f32_byte_size(n);
        total += bytes;
    }
    return total;
}

std::optional<LlamaLoadResult> load_llama_from_gguf(
        const std::string& path, std::ostream& diag) {
    diag << "loading GGUF: " << path << "\n";

    GgufFile gf;
    try {
        gf = GgufFile::open(path);
    } catch (const std::exception& e) {
        diag << "  - cannot open: " << e.what() << "\n";
        return std::nullopt;
    }
    diag << "  version=" << gf.version()
         << "  n_tensors=" << gf.n_tensors()
         << "  n_kv=" << gf.n_kv() << "\n";

    // ----- 1. Read LlamaConfig KV keys ---------------------------------
    LlamaConfig cfg;  // start with defaults
    bool ok = true;

    auto v = read_kv<uint32_t>(gf, "llama.vocab_size", as_uint32, diag, "uint32");
    if (!v) ok = false; else cfg.vocab_size = *v;

    v = read_kv<uint32_t>(gf, "llama.embedding_length", as_uint32, diag, "uint32");
    if (!v) ok = false; else cfg.hidden = *v;

    v = read_kv<uint32_t>(gf, "llama.feed_forward_length", as_uint32, diag, "uint32");
    if (!v) ok = false; else cfg.intermediate = *v;

    v = read_kv<uint32_t>(gf, "llama.attention.head_count", as_uint32, diag, "uint32");
    if (!v) ok = false; else cfg.n_heads = *v;

    v = read_kv<uint32_t>(gf, "llama.attention.head_count_kv", as_uint32, diag, "uint32");
    if (!v) ok = false; else cfg.n_kv_heads = *v;

    v = read_kv<uint32_t>(gf, "llama.block_count", as_uint32, diag, "uint32");
    if (!v) ok = false; else cfg.n_layers = *v;

    v = read_kv<uint32_t>(gf, "llama.context_length", as_uint32, diag, "uint32");
    if (!v) ok = false; else cfg.max_seq_len = *v;

    // Optional KVs — fall back to LlamaConfig defaults on absence.
    if (auto eps = read_kv<float>(gf, "llama.attention.layer_norm_rms_epsilon",
                                  as_float32, diag, "float32")) {
        cfg.rms_norm_eps = *eps;
    }
    if (auto theta = read_kv<float>(gf, "llama.rope.freq_base",
                                    as_float32, diag, "float32")) {
        cfg.theta_base = *theta;
    }

    // Derived: head_dim = hidden / n_heads.
    if (cfg.n_heads > 0 && cfg.hidden % cfg.n_heads == 0) {
        cfg.head_dim = cfg.hidden / cfg.n_heads;
    } else {
        diag << "  - embedding_length (" << cfg.hidden
             << ") must be divisible by head_count (" << cfg.n_heads << ")\n";
        ok = false;
    }

    if (cfg.n_kv_heads > cfg.n_heads) {
        diag << "  - head_count_kv (" << cfg.n_kv_heads
             << ") cannot exceed head_count (" << cfg.n_heads << ")\n";
        ok = false;
    }

    if (!ok) {
        diag << "  -> aborting: bad/missing metadata\n";
        return std::nullopt;
    }

    diag << "  cfg: vocab=" << cfg.vocab_size
         << " hidden=" << cfg.hidden
         << " intermediate=" << cfg.intermediate
         << " heads=" << cfg.n_heads
         << " kv_heads=" << cfg.n_kv_heads
         << " head_dim=" << cfg.head_dim
         << " layers=" << cfg.n_layers
         << " max_seq=" << cfg.max_seq_len
         << " rms_eps=" << cfg.rms_norm_eps
         << " theta=" << cfg.theta_base << "\n";

    // ----- 2. Pre-flight: confirm every named tensor exists. -----------
    auto must_find = [&](const std::string& name) -> std::optional<std::size_t> {
        auto idx = gf.find_tensor(name);
        if (!idx.has_value()) {
            diag << "  - missing tensor: " << name << "\n";
            ok = false;
        }
        return idx;
    };

    auto idx_embed = must_find("token_embd.weight");
    auto idx_norm  = must_find("output_norm.weight");
    auto idx_out   = gf.find_tensor("output.weight");  // optional
    if (idx_out.has_value()) {
        diag << "  output.weight present (un-tied)\n";
    } else {
        diag << "  output.weight absent; tying W_output = W_embed^T\n";
    }

    // Per-block indices, gathered upfront so a missing one doesn't leave
    // a half-populated weights blob.
    std::vector<std::optional<std::size_t>> idx_attn_norm(cfg.n_layers);
    std::vector<std::optional<std::size_t>> idx_attn_q(cfg.n_layers);
    std::vector<std::optional<std::size_t>> idx_attn_k(cfg.n_layers);
    std::vector<std::optional<std::size_t>> idx_attn_v(cfg.n_layers);
    std::vector<std::optional<std::size_t>> idx_attn_o(cfg.n_layers);
    std::vector<std::optional<std::size_t>> idx_mlp_norm(cfg.n_layers);
    std::vector<std::optional<std::size_t>> idx_mlp_gate(cfg.n_layers);
    std::vector<std::optional<std::size_t>> idx_mlp_up(cfg.n_layers);
    std::vector<std::optional<std::size_t>> idx_mlp_down(cfg.n_layers);

    for (int64_t i = 0; i < cfg.n_layers; ++i) {
        std::ostringstream p; p << "blk." << i << ".";
        const std::string pre = p.str();
        idx_attn_norm[i] = must_find(pre + "attn_norm.weight");
        idx_attn_q[i]    = must_find(pre + "attn_q.weight");
        idx_attn_k[i]    = must_find(pre + "attn_k.weight");
        idx_attn_v[i]    = must_find(pre + "attn_v.weight");
        idx_attn_o[i]    = must_find(pre + "attn_output.weight");
        idx_mlp_norm[i]  = must_find(pre + "ffn_norm.weight");
        idx_mlp_gate[i]  = must_find(pre + "ffn_gate.weight");
        idx_mlp_up[i]    = must_find(pre + "ffn_up.weight");
        idx_mlp_down[i]  = must_find(pre + "ffn_down.weight");
    }

    if (!ok) {
        diag << "  -> aborting: missing tensors\n";
        return std::nullopt;
    }

    // ----- 3. Soft-warn on big F32 footprints. -------------------------
    // Estimate F32-dequantized size by summing the on-disk byte sizes of
    // the tensors we just enumerated (Q4_0/Q8_0 inflate by ~7x, F16 by
    // 2x; we use the worst-case raw-byte * 4 as a uniform F32 estimate).
    std::size_t on_disk_bytes = 0;
    auto add_bytes = [&](std::optional<std::size_t> idx) {
        if (!idx.has_value()) return;
        const auto& ti = gf.tensor_infos()[*idx];
        int64_t n = 1;
        for (uint64_t d : ti.dims) n *= static_cast<int64_t>(d);
        std::size_t b = tensor_byte_size(n, ti.type);
        if (b > 0) on_disk_bytes += b;
    };
    add_bytes(idx_embed);
    add_bytes(idx_out);
    add_bytes(idx_norm);
    for (int64_t i = 0; i < cfg.n_layers; ++i) {
        add_bytes(idx_attn_norm[i]); add_bytes(idx_attn_q[i]);
        add_bytes(idx_attn_k[i]);    add_bytes(idx_attn_v[i]);
        add_bytes(idx_attn_o[i]);
        add_bytes(idx_mlp_norm[i]);  add_bytes(idx_mlp_gate[i]);
        add_bytes(idx_mlp_up[i]);    add_bytes(idx_mlp_down[i]);
    }
    std::size_t f32_bytes = on_disk_bytes * 4;  // upper bound (worst case)
    if (f32_bytes > kLlamaLoadSoftWarnBytes) {
        diag << "  WARNING: estimated F32 weight memory is "
             << (f32_bytes / (1024 * 1024)) << " MiB (>"
             << (kLlamaLoadSoftWarnBytes / (1024 * 1024)) << " MiB).\n"
             << "           This will materialize ~" << (f32_bytes / (1024*1024))
             << " MiB of float32 in RAM.\n"
             << "           Proceeding anyway (override kLlamaLoadSoftWarnBytes to silence).\n";
    } else {
        diag << "  F32 weight memory estimate: "
             << (f32_bytes / 1024) << " KiB (under threshold)\n";
    }

    // ----- 4. Load tensors. --------------------------------------------
    LlamaModelWeights w;
    auto load = [&](std::optional<std::size_t> idx,
                    const std::string& name) -> Tensor {
        return take_loaded(gf.load_tensor(*idx));
    };

    w.W_embed   = load(idx_embed, "token_embd.weight");
    check_shape(w.W_embed, {cfg.vocab_size, cfg.hidden},
                "token_embd.weight", diag);

    w.final_norm = load(idx_norm, "output_norm.weight");
    check_shape(w.final_norm, {cfg.hidden}, "output_norm.weight", diag);

    if (idx_out.has_value()) {
        w.W_output = load(idx_out, "output.weight");
        check_shape(w.W_output, {cfg.hidden, cfg.vocab_size},
                    "output.weight", diag);
    } else {
        // Tied embeddings: share the same storage as W_embed. We do this
        // by transposing at lookup time would be expensive; instead we
        // store a copy of the embedding rows. W_output is [hidden, vocab]
        // in our convention; W_embed is [vocab, hidden]. For tied models
        // (Llama 2/3), we copy the storage and transpose the view.
        Tensor tied = w.W_embed.transpose();
        // Ensure contiguous so LlamaModel can rely on stride semantics.
        w.W_output = tied.contiguous();
        diag << "  tied W_output = W_embed^T (copy)\n";
    }

    int64_t H  = cfg.hidden;
    int64_t I  = cfg.intermediate;
    int64_t Hd = cfg.head_dim;
    int64_t KH = cfg.n_kv_heads * Hd;
    int64_t QH = cfg.n_heads * Hd;

    w.blocks.resize(static_cast<std::size_t>(cfg.n_layers));
    for (int64_t i = 0; i < cfg.n_layers; ++i) {
        auto& bw = w.blocks[static_cast<std::size_t>(i)].weights;

        bw.attn_norm = load(idx_attn_norm[i], "blk.i.attn_norm.weight");
        check_shape(bw.attn_norm, {H}, "attn_norm", diag);

        bw.attn.Wq = load(idx_attn_q[i], "blk.i.attn_q.weight");
        check_shape(bw.attn.Wq, {H, QH}, "attn_q", diag);

        bw.attn.Wk = load(idx_attn_k[i], "blk.i.attn_k.weight");
        check_shape(bw.attn.Wk, {H, KH}, "attn_k", diag);

        bw.attn.Wv = load(idx_attn_v[i], "blk.i.attn_v.weight");
        check_shape(bw.attn.Wv, {H, KH}, "attn_v", diag);

        bw.attn.Wo = load(idx_attn_o[i], "blk.i.attn_output.weight");
        check_shape(bw.attn.Wo, {QH, H}, "attn_output", diag);

        bw.mlp_norm = load(idx_mlp_norm[i], "blk.i.ffn_norm.weight");
        check_shape(bw.mlp_norm, {H}, "ffn_norm", diag);

        bw.mlp.W_gate = load(idx_mlp_gate[i], "blk.i.ffn_gate.weight");
        check_shape(bw.mlp.W_gate, {H, I}, "ffn_gate", diag);

        bw.mlp.W_up = load(idx_mlp_up[i], "blk.i.ffn_up.weight");
        check_shape(bw.mlp.W_up, {H, I}, "ffn_up", diag);

        bw.mlp.W_down = load(idx_mlp_down[i], "blk.i.ffn_down.weight");
        check_shape(bw.mlp.W_down, {I, H}, "ffn_down", diag);
    }

    diag << "  loaded " << cfg.n_layers << " blocks; "
         << "W_embed=" << w.W_embed.shape()[0] << "x" << w.W_embed.shape()[1]
         << ", W_output=" << w.W_output.shape()[0] << "x" << w.W_output.shape()[1]
         << "\n";

    LlamaLoadResult result;
    result.cfg = cfg;
    result.weights = std::move(w);
    return result;
}

}  // namespace tinyllm