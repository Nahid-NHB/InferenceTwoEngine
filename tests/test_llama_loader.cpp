// tests/test_llama_loader.cpp
// -----------------------------------------------------------------------------
// Tests for the GGUF -> LlamaModel loader.
//
// We build a synthetic GGUF in /tmp using an extended Writer fixture (a
// minimal Llama-shaped file: vocab=32, hidden=8, n_heads=2, head_dim=4,
// n_kv_heads=2, intermediate=16, n_layers=2, all-F32 weights), then
// invoke `load_llama_from_gguf` and verify:
//   1. The returned LlamaConfig matches the file's metadata.
//   2. Every required tensor is found by name.
//   3. The loader cleanly reports missing tensors (missing-tensor error).
//   4. The loaded weights, fed through `make_model` + a forward pass,
//      produce logits identical to a hand-built reference model.
//   5. `estimate_llama_f32_bytes` gives the right order of magnitude.
// -----------------------------------------------------------------------------
#include "test_helpers.hpp"
#include "tinyllm/gguf.hpp"
#include "tinyllm/llama_block.hpp"
#include "tinyllm/llama_loader.hpp"
#include "tinyllm/model.hpp"
#include "tinyllm/sampler.hpp"
#include "tinyllm/tensor.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace tinyllm;

namespace {

// -----------------------------------------------------------------------------
// Writer fixture — extends the test_gguf.cpp one with the array and
// string helpers the loader test needs.
// -----------------------------------------------------------------------------
class Writer {
public:
    explicit Writer(std::ostream& o) : o_(o) {}

    template <typename T> void le(T v) {
        o_.write(reinterpret_cast<const char*>(&v), sizeof(T));
    }
    void bytes(const void* p, std::size_t n) { o_.write(reinterpret_cast<const char*>(p), n); }
    void str(std::string_view s) {
        le<uint64_t>(s.size());
        bytes(s.data(), s.size());
    }
    void value_uint32(uint32_t x) {
        le<uint32_t>(static_cast<uint32_t>(GgufValueType::Uint32));
        le<uint32_t>(x);
    }
    void value_float32(float x) {
        le<uint32_t>(static_cast<uint32_t>(GgufValueType::Float32));
        le<float>(x);
    }
    void value_string(std::string_view s) {
        le<uint32_t>(static_cast<uint32_t>(GgufValueType::String));
        str(s);
    }
    void value_array_string(const std::vector<std::string>& arr) {
        le<uint32_t>(static_cast<uint32_t>(GgufValueType::Array));
        le<uint32_t>(static_cast<uint32_t>(GgufValueType::String));
        le<uint64_t>(arr.size());
        for (const auto& s : arr) str(s);
    }
    void tensor_info(std::string_view name, const std::vector<uint64_t>& dims,
                     GgufTensorType dt, uint64_t offset) {
        str(name);
        le<uint32_t>(static_cast<uint32_t>(dims.size()));
        for (uint64_t d : dims) le<uint64_t>(d);
        le<uint32_t>(static_cast<uint32_t>(dt));
        le<uint64_t>(offset);
    }

private:
    std::ostream& o_;
};

// -----------------------------------------------------------------------------
// Hand-built reference weights: tiny deterministic shapes. Same set of
// values are written to the synthetic GGUF, so the loader should yield a
// LlamaLoadResult that produces the same logits.
// -----------------------------------------------------------------------------
struct RefShapes {
    int64_t vocab        = 32;
    int64_t hidden       = 8;
    int64_t intermediate = 16;
    int64_t n_heads      = 2;
    int64_t n_kv_heads   = 2;
    int64_t head_dim     = 4;
    int64_t n_layers     = 2;
    int64_t max_seq      = 64;
    float   rms_eps      = 1e-5f;
    float   theta_base   = 10000.0f;
};

// Deterministic LCG, identical to the one in generate_demo.cpp.
struct Lcg {
    uint64_t s;
    explicit Lcg(uint64_t seed) : s(seed ? seed : 1) {}
    float next_unit() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        uint32_t bits = static_cast<uint32_t>((s >> 11) & ((1ULL << 21) - 1));
        return (static_cast<float>(bits) / static_cast<float>(1ULL << 21)) * 0.1f - 0.05f;
    }
};

// Pad helpers.
static std::size_t pad_to(std::ostream& o, std::size_t pos, std::size_t kAlign) {
    while (pos % kAlign != 0) { o.put(0); ++pos; }
    return pos;
}

// Names of every required tensor (one per slot).
struct TensorSpec {
    std::string name;
    std::vector<int64_t> shape;
    std::vector<float> data;
};

static std::vector<TensorSpec> build_tensor_specs(const RefShapes& s) {
    std::vector<TensorSpec> out;
    Lcg rng(0xC0FFEE);

    auto emit = [&](const std::string& n, std::vector<int64_t> shape) {
        int64_t n_elts = 1;
        for (auto d : shape) n_elts *= d;
        std::vector<float> data(static_cast<std::size_t>(n_elts));
        for (auto& v : data) v = rng.next_unit();
        out.push_back({n, std::move(shape), std::move(data)});
    };

    emit("token_embd.weight",  {s.vocab, s.hidden});
    emit("output.weight",      {s.hidden, s.vocab});  // un-tied
    emit("output_norm.weight", {s.hidden});

    int64_t QH = s.n_heads * s.head_dim;
    int64_t KH = s.n_kv_heads * s.head_dim;
    for (int64_t i = 0; i < s.n_layers; ++i) {
        std::ostringstream p; p << "blk." << i << ".";
        const std::string pre = p.str();
        emit(pre + "attn_norm.weight",   {s.hidden});
        emit(pre + "attn_q.weight",      {s.hidden, QH});
        emit(pre + "attn_k.weight",      {s.hidden, KH});
        emit(pre + "attn_v.weight",      {s.hidden, KH});
        emit(pre + "attn_output.weight", {QH, s.hidden});
        emit(pre + "ffn_norm.weight",    {s.hidden});
        emit(pre + "ffn_gate.weight",    {s.hidden, s.intermediate});
        emit(pre + "ffn_up.weight",      {s.hidden, s.intermediate});
        emit(pre + "ffn_down.weight",    {s.intermediate, s.hidden});
    }
    return out;
}

struct SyntheticFile {
    std::string path;
    RefShapes shapes;
    std::vector<TensorSpec> specs;
};

static SyntheticFile write_synthetic_llama_gguf() {
    SyntheticFile f;
    f.path = "/tmp/tinyllm_test_loader.gguf";
    f.shapes = RefShapes{};
    f.specs  = build_tensor_specs(f.shapes);

    constexpr std::size_t kAlign = 32;
    std::ofstream out(f.path, std::ios::binary);
    Writer w(out);

    // Header.
    out.write("GGUF", 4);
    w.le<uint32_t>(3);
    w.le<uint64_t>(f.specs.size());
    // KV count below; we'll fill it in after writing the rest of the KVs.
    // For simplicity we hardcode the count (matches the file we emit).
    constexpr uint64_t kKvCount = 11;
    w.le<uint64_t>(kKvCount);

    // ---- Metadata KV pairs (Llama convention) ----
    w.str("general.architecture");                    w.value_string("llama");
    w.str("llama.vocab_size");                        w.value_uint32(static_cast<uint32_t>(f.shapes.vocab));
    w.str("llama.embedding_length");                  w.value_uint32(static_cast<uint32_t>(f.shapes.hidden));
    w.str("llama.feed_forward_length");               w.value_uint32(static_cast<uint32_t>(f.shapes.intermediate));
    w.str("llama.attention.head_count");              w.value_uint32(static_cast<uint32_t>(f.shapes.n_heads));
    w.str("llama.attention.head_count_kv");           w.value_uint32(static_cast<uint32_t>(f.shapes.n_kv_heads));
    w.str("llama.block_count");                       w.value_uint32(static_cast<uint32_t>(f.shapes.n_layers));
    w.str("llama.context_length");                    w.value_uint32(static_cast<uint32_t>(f.shapes.max_seq));
    w.str("llama.attention.layer_norm_rms_epsilon");  w.value_float32(f.shapes.rms_eps);
    w.str("llama.rope.freq_base");                    w.value_float32(f.shapes.theta_base);
    // Tokenizer-model KV (the loader doesn't need it, but real files have it).
    w.str("tokenizer.ggml.model");                    w.value_string("llama");

    // ---- Compute data-section offsets for each tensor ----
    std::vector<uint64_t> offsets;
    uint64_t cursor = 0;
    auto pad = [&]() { cursor = (cursor + kAlign - 1) & ~(kAlign - 1); };
    for (const auto& ts : f.specs) {
        pad();
        offsets.push_back(cursor);
        int64_t n_elts = 1;
        for (auto d : ts.shape) n_elts *= d;
        cursor += static_cast<uint64_t>(n_elts) * sizeof(float);
    }

    // ---- Tensor infos ----
    for (std::size_t i = 0; i < f.specs.size(); ++i) {
        std::vector<uint64_t> dims;
        for (auto d : f.specs[i].shape) dims.push_back(static_cast<uint64_t>(d));
        w.tensor_info(f.specs[i].name, dims, GgufTensorType::F32, offsets[i]);
    }

    // ---- Alignment (v3) ----
    w.le<uint64_t>(kAlign);

    // Pad file pos to alignment for the first tensor's data.
    auto pos = static_cast<std::uint64_t>(out.tellp());
    pos = pad_to(out, pos, kAlign);

    // ---- Data section ----
    for (std::size_t i = 0; i < f.specs.size(); ++i) {
        const auto& ts = f.specs[i];
        out.write(reinterpret_cast<const char*>(ts.data.data()),
                  ts.data.size() * sizeof(float));
        pos = static_cast<std::uint64_t>(out.tellp());
        pos = pad_to(out, pos, kAlign);
    }
    return f;
}

// Build a hand-built reference LlamaModel from the same tensor specs.
// Result should produce identical logits to one built from the loaded
// weights.
static LlamaModel build_reference_model(const SyntheticFile& sf) {
    auto find = [&](const std::string& n) -> const TensorSpec* {
        for (const auto& s : sf.specs) if (s.name == n) return &s;
        return nullptr;
    };

    LlamaConfig cfg;
    cfg.vocab_size   = sf.shapes.vocab;
    cfg.hidden       = sf.shapes.hidden;
    cfg.intermediate = sf.shapes.intermediate;
    cfg.n_heads      = sf.shapes.n_heads;
    cfg.n_kv_heads   = sf.shapes.n_kv_heads;
    cfg.head_dim     = sf.shapes.head_dim;
    cfg.n_layers     = sf.shapes.n_layers;
    cfg.max_seq_len  = sf.shapes.max_seq;
    cfg.rms_norm_eps = sf.shapes.rms_eps;
    cfg.theta_base   = sf.shapes.theta_base;

    auto to_tensor = [](const TensorSpec& ts) {
        return Tensor(ts.shape, DType::Float32, ts.data.data(),
                      ts.data.size() * sizeof(float));
    };

    LlamaModelWeights w;
    w.W_embed    = to_tensor(*find("token_embd.weight"));
    w.W_output   = to_tensor(*find("output.weight"));
    w.final_norm = to_tensor(*find("output_norm.weight"));

    int64_t QH = cfg.n_heads * cfg.head_dim;
    int64_t KH = cfg.n_kv_heads * cfg.head_dim;
    w.blocks.resize(static_cast<std::size_t>(cfg.n_layers));
    for (int64_t i = 0; i < cfg.n_layers; ++i) {
        std::ostringstream p; p << "blk." << i << ".";
        const std::string pre = p.str();
        auto& bw = w.blocks[static_cast<std::size_t>(i)].weights;
        bw.attn_norm = to_tensor(*find(pre + "attn_norm.weight"));
        bw.attn.Wq   = to_tensor(*find(pre + "attn_q.weight"));
        bw.attn.Wk   = to_tensor(*find(pre + "attn_k.weight"));
        bw.attn.Wv   = to_tensor(*find(pre + "attn_v.weight"));
        bw.attn.Wo   = to_tensor(*find(pre + "attn_output.weight"));
        bw.mlp_norm  = to_tensor(*find(pre + "ffn_norm.weight"));
        bw.mlp.W_gate = to_tensor(*find(pre + "ffn_gate.weight"));
        bw.mlp.W_up   = to_tensor(*find(pre + "ffn_up.weight"));
        bw.mlp.W_down = to_tensor(*find(pre + "ffn_down.weight"));
    }
    return make_model(std::move(w), cfg);
}

// Dump the synthetic file to a known path so the driver can pick it up.
static void dump_fixture_for_driver() {
    auto f = write_synthetic_llama_gguf();
    std::printf("[fixture] wrote synthetic Llama GGUF to %s\n", f.path.c_str());
}

}  // namespace

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------
TEST_CASE(loader_populates_config) {
    auto sf = write_synthetic_llama_gguf();
    std::ostringstream diag;
    auto result = load_llama_from_gguf(sf.path, diag);
    REQUIRE(result.has_value());
    REQUIRE(result->cfg.vocab_size   == sf.shapes.vocab);
    REQUIRE(result->cfg.hidden       == sf.shapes.hidden);
    REQUIRE(result->cfg.intermediate == sf.shapes.intermediate);
    REQUIRE(result->cfg.n_heads      == sf.shapes.n_heads);
    REQUIRE(result->cfg.n_kv_heads   == sf.shapes.n_kv_heads);
    REQUIRE(result->cfg.head_dim     == sf.shapes.head_dim);
    REQUIRE(result->cfg.n_layers     == sf.shapes.n_layers);
    REQUIRE(result->cfg.max_seq_len  == sf.shapes.max_seq);
    REQUIRE_NEAR(result->cfg.rms_norm_eps, sf.shapes.rms_eps, 1e-9);
    REQUIRE_NEAR(result->cfg.theta_base,   sf.shapes.theta_base, 1e-9);
}

TEST_CASE(loader_finds_every_named_tensor) {
    auto sf = write_synthetic_llama_gguf();
    std::ostringstream diag;
    auto result = load_llama_from_gguf(sf.path, diag);
    REQUIRE(result.has_value());

    auto& w = result->weights;
    REQUIRE(w.W_embed.shape()[0]    == sf.shapes.vocab);
    REQUIRE(w.W_embed.shape()[1]    == sf.shapes.hidden);
    REQUIRE(w.W_output.shape()[0]   == sf.shapes.hidden);
    REQUIRE(w.W_output.shape()[1]   == sf.shapes.vocab);
    REQUIRE(w.final_norm.shape()[0] == sf.shapes.hidden);
    REQUIRE(w.blocks.size() == static_cast<std::size_t>(sf.shapes.n_layers));

    int64_t QH = sf.shapes.n_heads * sf.shapes.head_dim;
    int64_t KH = sf.shapes.n_kv_heads * sf.shapes.head_dim;
    for (const auto& b : w.blocks) {
        const auto& bw = b.weights;
        REQUIRE(bw.attn_norm.shape()[0] == sf.shapes.hidden);
        REQUIRE(bw.attn.Wq.shape()[0] == sf.shapes.hidden);
        REQUIRE(bw.attn.Wq.shape()[1] == QH);
        REQUIRE(bw.attn.Wk.shape()[1] == KH);
        REQUIRE(bw.attn.Wv.shape()[1] == KH);
        REQUIRE(bw.attn.Wo.shape()[0] == QH);
        REQUIRE(bw.attn.Wo.shape()[1] == sf.shapes.hidden);
        REQUIRE(bw.mlp.W_gate.shape()[1] == sf.shapes.intermediate);
        REQUIRE(bw.mlp.W_down.shape()[0] == sf.shapes.intermediate);
        REQUIRE(bw.mlp.W_down.shape()[1] == sf.shapes.hidden);
    }
}

TEST_CASE(loader_fails_cleanly_on_missing_tensor) {
    // Write the synthetic file then delete one of the block tensors on disk
    // by re-writing the header with one fewer tensor info entry. Simpler:
    // write a GGUF with the full metadata but only the embedding tensor
    // (everything else is omitted). The loader should report missing
    // tensors and return nullopt.
    std::string path = "/tmp/tinyllm_test_loader_missing.gguf";
    {
        std::ofstream out(path, std::ios::binary);
        Writer w(out);
        out.write("GGUF", 4);
        w.le<uint32_t>(3);
        w.le<uint64_t>(1);                  // n_tensors = 1 (just embed)
        w.le<uint64_t>(8);                  // n_kv
        w.str("general.architecture");                    w.value_string("llama");
        w.str("llama.vocab_size");                        w.value_uint32(32);
        w.str("llama.embedding_length");                  w.value_uint32(8);
        w.str("llama.feed_forward_length");               w.value_uint32(16);
        w.str("llama.attention.head_count");              w.value_uint32(2);
        w.str("llama.attention.head_count_kv");           w.value_uint32(2);
        w.str("llama.block_count");                       w.value_uint32(2);
        w.str("llama.context_length");                    w.value_uint32(64);
        // No rms_eps / theta_base on purpose.

        // Tensor info: only embed.
        std::vector<float> embed(32 * 8, 0.0f);
        w.tensor_info("token_embd.weight", {32, 8}, GgufTensorType::F32, 0);
        w.le<uint64_t>(32);  // alignment

        auto pos = static_cast<std::uint64_t>(out.tellp());
        while (pos % 32 != 0) { out.put(0); ++pos; }
        out.write(reinterpret_cast<const char*>(embed.data()),
                  embed.size() * sizeof(float));
    }

    std::ostringstream diag;
    auto result = load_llama_from_gguf(path, diag);
    REQUIRE(!result.has_value());
    // The diag should mention the first missing tensor.
    REQUIRE(diag.str().find("missing tensor") != std::string::npos);
}

TEST_CASE(loader_round_trips_a_real_shape_model) {
    auto sf = write_synthetic_llama_gguf();
    std::ostringstream diag;
    auto loaded = load_llama_from_gguf(sf.path, diag);
    REQUIRE(loaded.has_value());

    LlamaModel m_loaded = make_model(std::move(loaded->weights), loaded->cfg);
    LlamaModel m_ref    = build_reference_model(sf);

    // Same tokens -> same logits.
    std::vector<int32_t> toks = {1, 5, 17, 23};
    Tensor t({static_cast<int64_t>(toks.size())}, DType::Int32);
    for (std::size_t i = 0; i < toks.size(); ++i) t.at_flat_int(static_cast<int64_t>(i)) = toks[i];

    Tensor logits_loaded = m_loaded.forward(t, 0);
    Tensor logits_ref    = m_ref.forward(t, 0);

    REQUIRE(logits_loaded.shape() == logits_ref.shape());
    REQUIRE(logits_loaded.dtype() == DType::Float32);
    for (int64_t i = 0; i < logits_loaded.numel(); ++i) {
        REQUIRE_NEAR(logits_loaded.at_flat(i), logits_ref.at_flat(i), 1e-5);
    }
}

TEST_CASE(loader_estimates_f32_memory) {
    auto sf = write_synthetic_llama_gguf();
    auto gf = GgufFile::open(sf.path);
    std::size_t bytes = estimate_llama_f32_bytes(gf);

    // Compute expected F32 size from the tensor specs we wrote.
    int64_t total_elts = 0;
    for (const auto& ts : sf.specs) {
        int64_t n = 1; for (auto d : ts.shape) n *= d;
        total_elts += n;
    }
    std::size_t expected = static_cast<std::size_t>(total_elts) * 4;
    REQUIRE(bytes == expected);

    // The synthetic file should be FAR below the 4 GiB soft warn.
    REQUIRE(bytes < kLlamaLoadSoftWarnBytes);
}

TEST_CASE(loader_handles_tied_embeddings) {
    // Build a file with no output.weight -> loader ties W_output = W_embed^T.
    std::string path = "/tmp/tinyllm_test_loader_tied.gguf";
    constexpr std::size_t kAlign = 32;
    RefShapes s;  // defaults: vocab=32, hidden=8, layers=2
    Lcg rng(0xBEEF);
    std::vector<TensorSpec> specs;
    auto emit = [&](const std::string& n, std::vector<int64_t> shape) {
        int64_t n_elts = 1;
        for (auto d : shape) n_elts *= d;
        std::vector<float> data(static_cast<std::size_t>(n_elts));
        for (auto& v : data) v = rng.next_unit();
        specs.push_back({n, std::move(shape), std::move(data)});
    };
    emit("token_embd.weight",  {s.vocab, s.hidden});
    emit("output_norm.weight", {s.hidden});
    int64_t QH = s.n_heads * s.head_dim;
    int64_t KH = s.n_kv_heads * s.head_dim;
    for (int64_t i = 0; i < s.n_layers; ++i) {
        std::ostringstream p; p << "blk." << i << ".";
        const std::string pre = p.str();
        emit(pre + "attn_norm.weight",   {s.hidden});
        emit(pre + "attn_q.weight",      {s.hidden, QH});
        emit(pre + "attn_k.weight",      {s.hidden, KH});
        emit(pre + "attn_v.weight",      {s.hidden, KH});
        emit(pre + "attn_output.weight", {QH, s.hidden});
        emit(pre + "ffn_norm.weight",    {s.hidden});
        emit(pre + "ffn_gate.weight",    {s.hidden, s.intermediate});
        emit(pre + "ffn_up.weight",      {s.hidden, s.intermediate});
        emit(pre + "ffn_down.weight",    {s.intermediate, s.hidden});
    }

    {
        std::ofstream out(path, std::ios::binary);
        Writer w(out);
        out.write("GGUF", 4);
        w.le<uint32_t>(3);
        w.le<uint64_t>(specs.size());
        w.le<uint64_t>(8);
        w.str("general.architecture");          w.value_string("llama");
        w.str("llama.vocab_size");              w.value_uint32(static_cast<uint32_t>(s.vocab));
        w.str("llama.embedding_length");        w.value_uint32(static_cast<uint32_t>(s.hidden));
        w.str("llama.feed_forward_length");     w.value_uint32(static_cast<uint32_t>(s.intermediate));
        w.str("llama.attention.head_count");    w.value_uint32(static_cast<uint32_t>(s.n_heads));
        w.str("llama.attention.head_count_kv"); w.value_uint32(static_cast<uint32_t>(s.n_kv_heads));
        w.str("llama.block_count");             w.value_uint32(static_cast<uint32_t>(s.n_layers));
        w.str("llama.context_length");          w.value_uint32(static_cast<uint32_t>(s.max_seq));

        std::vector<uint64_t> offsets;
        uint64_t cursor = 0;
        auto pad = [&]() { cursor = (cursor + kAlign - 1) & ~(kAlign - 1); };
        for (const auto& ts : specs) {
            pad();
            offsets.push_back(cursor);
            int64_t n_elts = 1; for (auto d : ts.shape) n_elts *= d;
            cursor += static_cast<uint64_t>(n_elts) * sizeof(float);
        }
        for (std::size_t i = 0; i < specs.size(); ++i) {
            std::vector<uint64_t> dims;
            for (auto d : specs[i].shape) dims.push_back(static_cast<uint64_t>(d));
            w.tensor_info(specs[i].name, dims, GgufTensorType::F32, offsets[i]);
        }
        w.le<uint64_t>(kAlign);
        auto pos = static_cast<std::uint64_t>(out.tellp());
        while (pos % kAlign != 0) { out.put(0); ++pos; }
        for (const auto& ts : specs) {
            out.write(reinterpret_cast<const char*>(ts.data.data()),
                      ts.data.size() * sizeof(float));
            pos = static_cast<std::uint64_t>(out.tellp());
            while (pos % kAlign != 0) { out.put(0); ++pos; }
        }
    }

    std::ostringstream diag;
    auto result = load_llama_from_gguf(path, diag);
    REQUIRE(result.has_value());
    REQUIRE(result->weights.W_output.shape()[0] == s.hidden);
    REQUIRE(result->weights.W_output.shape()[1] == s.vocab);
    REQUIRE(diag.str().find("tying W_output") != std::string::npos);
}

// Write the synthetic fixture file (used by hand-running the driver).
TEST_CASE(loader_writes_fixture_for_driver) {
    dump_fixture_for_driver();
}