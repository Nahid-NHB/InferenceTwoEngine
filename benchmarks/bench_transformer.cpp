// benchmarks/bench_transformer.cpp
// -----------------------------------------------------------------------------
// Phase 10 unified benchmarking suite.
//
// Measures wall-time for every kernel that matters in a Llama-style
// forward pass, at sizes that match the two configurations we care
// about for this project:
//
//   "small"  : hidden=512,  heads=8,  head_dim=64, intermediate=1024
//   "medium" : hidden=1024, heads=8,  head_dim=128, intermediate=2048
//   "llama"  : hidden=2048, heads=16, head_dim=128, intermediate=5504
//
// Sizes chosen so the medium config fits in cache comfortably and the
// "llama" config approximates a single Llama-2-7B layer.
//
// Kernels covered:
//
//   Tensor elementwise:   add, mul_scalar, mul, sum, mean, softmax
//   Tensor shape ops:     reshape, transpose
//   Layer-norm:           rmsnorm
//   RoPE:                 rope_inplace
//   Attention:            attention_forward (prefill) + cached decode
//   MLP:                  mlp_forward (SwiGLU)
//   Llama block:          llama_block_forward + cached
//   Quantized matvec:     matmul_q4_0_f32 (AVX2 fused)
//   End-to-end:           llama_forward / cached
//
// Output is human-readable text tables + a CSV block at the bottom for
// scripted processing. Each measurement reports ms-per-call, calls/sec,
// and (where applicable) GB/s.
//
// Use the `--quick` flag to limit iterations.
// -----------------------------------------------------------------------------
#include "tinyllm/attention.hpp"
#include "tinyllm/llama_block.hpp"
#include "tinyllm/matmul.hpp"
#include "tinyllm/mlp.hpp"
#include "tinyllm/model.hpp"
#include "tinyllm/quantize.hpp"
#include "tinyllm/rmsnorm.hpp"
#include "tinyllm/rope.hpp"
#include "tinyllm/tensor.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace tinyllm;
using clk = std::chrono::high_resolution_clock;

// -----------------------------------------------------------------------------
// Small helpers
// -----------------------------------------------------------------------------
static std::uniform_real_distribution<float> bench_dist(-1.0f, 1.0f);
// Token ids use the per-config vocab_size; the bench_id() lambda below
// is rebuilt inside the per-config loop.
static std::uniform_int_distribution<int32_t> bench_id(0, 1);

struct BenchResult {
    std::string name;
    std::string shape;
    double      ms;
    double      units_per_sec;  // whatever the unit is (calls, elements, GB/s)
    std::string unit;
};

static std::vector<float> rand_vec(std::size_t n, std::mt19937& rng) {
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

static Tensor make_tensor(std::vector<int64_t> shape,
                          std::mt19937& rng) {
    std::size_t n = 1;
    for (auto d : shape) n *= static_cast<std::size_t>(d);
    auto v = rand_vec(n, rng);
    return Tensor(shape, DType::Float32, v.data(), n * sizeof(float));
}

template <typename Fn>
static BenchResult bench(const std::string& name,
                         const std::string& shape,
                         int iters,
                         Fn&& f) {
    // Warmup
    auto warm = f();
    (void)warm;
    auto t0 = clk::now();
    for (int i = 0; i < iters; ++i) {
        auto r = f();
        (void)r;
    }
    auto t1 = clk::now();
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double ms_per = total_ms / static_cast<double>(iters);
    return {name, shape, ms_per,
            1000.0 / ms_per, "calls/s"};
}

// -----------------------------------------------------------------------------
// Synthetic model factories. We build a LlamaModelWeights blob with
// randomly-initialized tensors, then wrap it in a LlamaModel so we can
// time the cached path too.
// -----------------------------------------------------------------------------
struct ModelCfg {
    std::string label;
    int64_t hidden;
    int64_t intermediate;
    int64_t n_heads;
    int64_t head_dim;       // = hidden / n_heads
    int64_t n_kv_heads;
    int64_t n_layers;
    int64_t vocab;
};

static LlamaModel make_random_model(const ModelCfg& mc, std::mt19937& rng) {
    LlamaConfig cfg;
    cfg.vocab_size   = mc.vocab;
    cfg.hidden       = mc.hidden;
    cfg.intermediate = mc.intermediate;
    cfg.n_heads      = mc.n_heads;
    cfg.n_kv_heads   = mc.n_kv_heads;
    cfg.head_dim     = mc.head_dim;
    cfg.n_layers     = mc.n_layers;

    auto fill = [&](std::vector<int64_t> shape) {
        return make_tensor(shape, rng);
    };
    LlamaModelWeights w;
    w.W_embed   = fill({mc.vocab, mc.hidden});
    w.final_norm = fill({mc.hidden});
    w.W_output  = fill({mc.hidden, mc.vocab});
    w.blocks.resize(static_cast<std::size_t>(mc.n_layers));
    for (auto& b : w.blocks) {
        b.weights.attn_norm = fill({mc.hidden});
        b.weights.mlp_norm  = fill({mc.hidden});
        int64_t qsz = mc.n_heads    * mc.head_dim;
        int64_t ksz = mc.n_kv_heads * mc.head_dim;
        b.weights.attn.Wq = fill({mc.hidden, qsz});
        b.weights.attn.Wk = fill({mc.hidden, ksz});
        b.weights.attn.Wv = fill({mc.hidden, ksz});
        b.weights.attn.Wo = fill({qsz, mc.hidden});
        b.weights.mlp.W_gate = fill({mc.hidden, mc.intermediate});
        b.weights.mlp.W_up   = fill({mc.hidden, mc.intermediate});
        b.weights.mlp.W_down = fill({mc.intermediate, mc.hidden});
    }
    return make_model(std::move(w), cfg);
}

static AttentionConfig attn_cfg_from(const ModelCfg& mc) {
    AttentionConfig cfg;
    cfg.hidden    = mc.hidden;
    cfg.n_heads   = mc.n_heads;
    cfg.n_kv_heads = mc.n_kv_heads;
    cfg.head_dim  = mc.head_dim;
    return cfg;
}

// -----------------------------------------------------------------------------
// Benchmark blocks, organized by kernel class.
// -----------------------------------------------------------------------------
static void bench_tensor_ops(std::vector<BenchResult>& out,
                             std::mt19937& rng, int iters) {
    auto add_a = make_tensor({1024, 1024}, rng);
    auto add_b = make_tensor({1024, 1024}, rng);
    out.push_back(bench("add", "1024x1024", iters, [&] {
        return ops::add(add_a, add_b);
    }));

    auto sm_in = make_tensor({32, 1024}, rng);
    out.push_back(bench("softmax", "32x1024", iters, [&] {
        return ops::softmax(sm_in);
    }));

    auto big = make_tensor({2048, 2048}, rng);
    out.push_back(bench("mul_scalar", "2048x2048", iters, [&] {
        return ops::mul_scalar(big, 0.5f);
    }));

    auto sa = make_tensor({512, 512}, rng);
    auto sb = make_tensor({512, 512}, rng);
    out.push_back(bench("multiply", "512x512", iters, [&] {
        return ops::multiply(sa, sb);
    }));

    auto sum_in = make_tensor({1024, 1024}, rng);
    out.push_back(bench("sum",      "1024x1024", iters, [&] {
        return ops::sum(sum_in);
    }));
    out.push_back(bench("mean",     "1024x1024", iters, [&] {
        return ops::mean(sum_in);
    }));

    auto r1 = make_tensor({512, 512}, rng);
    out.push_back(bench("reshape",  "{512,512}->{1024,256}", iters, [&] {
        return r1.reshape({1024, 256});
    }));

    auto t1 = make_tensor({64, 128, 256}, rng);
    out.push_back(bench("transpose", "{64,128,256} (permuted)", iters, [&] {
        return t1.transpose({0, 2, 1});
    }));
}

static void bench_matmul(std::vector<BenchResult>& out,
                         std::mt19937& rng, int iters) {
    for (int64_t n : {256, 1024, 2048}) {
        std::vector<float> a(static_cast<std::size_t>(n * n));
        std::vector<float> b(static_cast<std::size_t>(n * n));
        std::uniform_real_distribution<float> d(-1.0f, 1.0f);
        for (auto& v : a) v = d(rng);
        for (auto& v : b) v = d(rng);
        Tensor A({n, n}, DType::Float32, a.data(), a.size() * sizeof(float));
        Tensor B({n, n}, DType::Float32, b.data(), b.size() * sizeof(float));
        char shape[64];
        std::snprintf(shape, sizeof(shape), "%ldx%ldx%ld", n, n, n);

        out.push_back(bench("matmul_avx2",     shape, iters, [&] {
            return ops::matmul(A, B, ops::MatmulVariant::Avx2);
        }));
        out.push_back(bench("matmul_blocked",  shape, iters, [&] {
            return ops::matmul(A, B, ops::MatmulVariant::Blocked);
        }));
        out.push_back(bench("matmul_threaded", shape, iters, [&] {
            return ops::matmul(A, B, ops::MatmulVariant::Threaded);
        }));
    }
}

static void bench_quantize_matvec(std::vector<BenchResult>& out,
                                  std::mt19937& rng, int iters) {
    for (auto [M, K] : std::vector<std::pair<int64_t, int64_t>>{
             {512, 4096}, {1024, 4096}, {2048, 4096}}) {
        std::size_t mk = static_cast<std::size_t>(M * K);
        std::vector<float> w(mk), x(static_cast<std::size_t>(K));
        std::uniform_real_distribution<float> d(-1.0f, 1.0f);
        for (auto& v : w) v = d(rng);
        for (auto& v : x) v = d(rng);

        auto packed = quantize_q4_0(w.data(), M * K);
        std::vector<float> y(static_cast<std::size_t>(M));
        char shape[32];
        std::snprintf(shape, sizeof(shape), "%ldx%ld", M, K);

        out.push_back(bench("matmul_q4_0_f32", shape, iters, [&] {
            matmul_q4_0_f32(packed.data(), M, K, x.data(), y.data());
            return y;
        }));
    }
}

static void bench_layer_ops(std::vector<BenchResult>& out,
                            std::mt19937& rng, int iters) {
    // rmsnorm on a [seq, hidden] tensor with hidden=1024.
    auto r_x = make_tensor({32, 1024}, rng);
    auto r_g = make_tensor({1024}, rng);
    out.push_back(bench("rmsnorm", "{32,1024}", iters, [&] {
        return rmsnorm(r_x, r_g);
    }));

    // RoPE on a [seq, n_heads, head_dim] tensor.
    int64_t n_heads = 8, head_dim = 128, seq = 32;
    Tensor r_in({seq, n_heads, head_dim}, DType::Float32);
    for (int64_t i = 0; i < r_in.numel(); ++i) r_in.at_flat(i) = bench_dist(rng);
    out.push_back(bench("rope_inplace", "{32,8,128}", iters, [&] {
        Tensor local = r_in;
        rope_inplace(local, /*start_pos=*/0);
        return local;
    }));
}

static void bench_attention(std::vector<BenchResult>& out,
                            const ModelCfg& mc, std::mt19937& rng,
                            int iters) {
    auto model = make_random_model(mc, rng);
    auto attn_cfg = attn_cfg_from(mc);

    // Prefill: seq=32.
    int64_t seq_prefill = 32;
    auto x = make_tensor({seq_prefill, mc.hidden}, rng);
    const auto& bw = model.weights.blocks[0].weights.attn;
    char shape_pre[48];
    std::snprintf(shape_pre, sizeof(shape_pre),
                  "prefill seq=%ld heads=%ld", seq_prefill, mc.n_heads);
    out.push_back(bench("attention_prefill", shape_pre, iters, [&] {
        return attention_forward(x, bw, attn_cfg, /*start_pos=*/0);
    }));

    // Decode: seq=1, full cache up to 64.
    KvCache cache(/*max_seq=*/64, mc.n_kv_heads, mc.head_dim);
    int64_t seq_decode = 1;
    auto xd = make_tensor({seq_decode, mc.hidden}, rng);
    // Pre-fill cache to start position 63 so the decode path sees a
    // realistic 64-token history.
    auto xwarm = make_tensor({63, mc.hidden}, rng);
    attention_forward_cached(xwarm, bw, attn_cfg, cache, 0);

    char shape_dec[48];
    std::snprintf(shape_dec, sizeof(shape_dec),
                  "decode seq=1 ctx=64 heads=%ld", mc.n_heads);
    out.push_back(bench("attention_decode", shape_dec, iters, [&] {
        // Reset cache to its 63-row warm state before each timed call,
        // otherwise the cache fills up by iters.
        cache.length = 63;
        return attention_forward_cached(xd, bw, attn_cfg, cache, 63);
    }));
}

static void bench_mlp(std::vector<BenchResult>& out,
                     const ModelCfg& mc, std::mt19937& rng, int iters) {
    int64_t seq = 32;
    auto x = make_tensor({seq, mc.hidden}, rng);
    MlpWeights mw;
    mw.W_gate = make_tensor({mc.hidden, mc.intermediate}, rng);
    mw.W_up   = make_tensor({mc.hidden, mc.intermediate}, rng);
    mw.W_down = make_tensor({mc.intermediate, mc.hidden}, rng);
    char shape[48];
    std::snprintf(shape, sizeof(shape), "seq=%ld h=%ld i=%ld",
                  seq, mc.hidden, mc.intermediate);
    out.push_back(bench("mlp_forward", shape, iters, [&] {
        return mlp_forward(x, mw);
    }));
}

static void bench_block(std::vector<BenchResult>& out,
                        const ModelCfg& mc, std::mt19937& rng, int iters) {
    auto model = make_random_model(mc, rng);
    auto attn_cfg = attn_cfg_from(mc);

    // Prefill.
    int64_t seq = 32;
    auto x = make_tensor({seq, mc.hidden}, rng);
    char shape_pre[48];
    std::snprintf(shape_pre, sizeof(shape_pre),
                  "prefill seq=%ld layers=1", seq);
    out.push_back(bench("block_prefill", shape_pre, iters, [&] {
        return llama_block_forward(x, model.weights.blocks[0].weights,
                                   attn_cfg, 0);
    }));

    // Decode with cache.
    KvCache cache(/*max_seq=*/64, mc.n_kv_heads, mc.head_dim);
    auto xwarm = make_tensor({63, mc.hidden}, rng);
    llama_block_forward_cached(xwarm, model.weights.blocks[0].weights,
                               attn_cfg, cache, 0);
    auto xd = make_tensor({1, mc.hidden}, rng);
    char shape_dec[48];
    std::snprintf(shape_dec, sizeof(shape_dec),
                  "decode seq=1 ctx=64 layers=1");
    out.push_back(bench("block_decode", shape_dec, iters, [&] {
        cache.length = 63;
        return llama_block_forward_cached(xd, model.weights.blocks[0].weights,
                                          attn_cfg, cache, 63);
    }));
}

static void bench_end_to_end(std::vector<BenchResult>& out,
                             const ModelCfg& mc, std::mt19937& rng,
                             int iters) {
    auto model = make_random_model(mc, rng);

    // Prefill
    int64_t seq = 32;
    std::vector<int32_t> tokens(static_cast<std::size_t>(seq));
    std::uniform_int_distribution<int32_t> id(0, static_cast<int32_t>(mc.vocab - 1));
    for (auto& t : tokens) t = id(rng);
    Tensor ttok({seq}, DType::Int32, tokens.data(), tokens.size() * sizeof(int32_t));

    char shape_pre[64];
    std::snprintf(shape_pre, sizeof(shape_pre),
                  "prefill seq=%ld L=%ld", seq, mc.n_layers);
    out.push_back(bench("llama_forward", shape_pre, iters, [&] {
        model.reset_caches();
        return model.forward(ttok, 0);
    }));

    // Cached: prefill once, then decode token-by-token (we measure
    // just the decode step after the initial prefill is done).
    std::vector<int32_t> one_token = {tokens[0]};
    Tensor single({1}, DType::Int32, one_token.data(), sizeof(int32_t));
    // Run prefill first to populate caches.
    model.forward(ttok, 0);

    char shape_dec[64];
    std::snprintf(shape_dec, sizeof(shape_dec),
                  "decode seq=1 ctx=32 L=%ld", mc.n_layers);
    out.push_back(bench("llama_forward_cached", shape_dec, iters, [&] {
        return model.forward_cached(single, seq);
    }));
}

// -----------------------------------------------------------------------------
// Pretty-printing
// -----------------------------------------------------------------------------
static void print_table(const std::vector<BenchResult>& rows,
                        const std::string& title) {
    std::printf("\n=== %s ===\n", title.c_str());
    std::printf("%-22s  %-26s  %10s  %14s\n",
                "kernel", "shape", "ms/call", "throughput");
    std::printf("%-22s  %-26s  %10s  %14s\n",
                "----------------------",
                "--------------------------",
                "----------",
        "--------------");
    for (const auto& r : rows) {
        std::printf("%-22s  %-26s  %10.3f  %10.1f %s\n",
                    r.name.c_str(), r.shape.c_str(), r.ms,
                    r.units_per_sec, r.unit.c_str());
    }
}

static void print_csv(const std::vector<BenchResult>& rows) {
    std::printf("\n# CSV\n");
    std::printf("kernel,shape,ms_per_call\n");
    for (const auto& r : rows) {
        std::printf("%s,%s,%.6f\n",
                    r.name.c_str(), r.shape.c_str(), r.ms);
    }
}

int main(int argc, char** argv) {
    int iters = 30;
    bool quick = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--quick") == 0) { quick = true; iters = 5; }
        else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            iters = std::atoi(argv[++i]);
        }
    }
    std::mt19937 rng(0xC0FFEE);

    ModelCfg small  {"small",  512,  1024,  8,  64,  8,  2, 4096};
    ModelCfg medium {"medium", 1024, 2048,  8,  128, 8,  2, 4096};
    ModelCfg big    {"big",    2048, 5504, 16, 128, 16, 2, 4096};

    std::printf("Phase 10 unified benchmark suite (iters=%d)\n", iters);
    std::printf("Auto-detected CPU: AVX2=%s, threads=%d\n",
                ops::have_avx2() ? "yes" : "no",
                ops::hardware_threads());

    std::vector<BenchResult> rows;

    // ---- Tensor ops (size-independent) ----
    {
        std::vector<BenchResult> t;
        bench_tensor_ops(t, rng, iters);
        print_table(t, "Tensor ops");
        rows.insert(rows.end(), t.begin(), t.end());
    }

    // ---- Matmul ----
    {
        std::vector<BenchResult> t;
        bench_matmul(t, rng, iters);
        print_table(t, "Matmul (F32)");
        rows.insert(rows.end(), t.begin(), t.end());
    }

    // ---- Quantized matvec ----
    {
        std::vector<BenchResult> t;
        bench_quantize_matvec(t, rng, iters);
        print_table(t, "Quantized matvec (Q4_0 x F32, AVX2 fused)");
        rows.insert(rows.end(), t.begin(), t.end());
    }

    // ---- Layer ops + attention + MLP + block per config ----
    for (const ModelCfg& mc : {small, medium, /* big skipped if quick */ quick ? small : big}) {
        char title[64];
        std::snprintf(title, sizeof(title), "Config: %s (hidden=%ld, L=%ld)",
                      mc.label.c_str(), mc.hidden, mc.n_layers);

        std::vector<BenchResult> t;
        {
            std::vector<BenchResult> tmp;
            bench_layer_ops(tmp, rng, iters);
            t.insert(t.end(), tmp.begin(), tmp.end());
        }
        {
            std::vector<BenchResult> tmp;
            bench_attention(tmp, mc, rng, iters);
            t.insert(t.end(), tmp.begin(), tmp.end());
        }
        {
            std::vector<BenchResult> tmp;
            bench_mlp(tmp, mc, rng, iters);
            t.insert(t.end(), tmp.begin(), tmp.end());
        }
        {
            std::vector<BenchResult> tmp;
            bench_block(tmp, mc, rng, iters);
            t.insert(t.end(), tmp.begin(), tmp.end());
        }
        {
            std::vector<BenchResult> tmp;
            bench_end_to_end(tmp, mc, rng, iters);
            t.insert(t.end(), tmp.begin(), tmp.end());
        }
        print_table(t, title);
        rows.insert(rows.end(), t.begin(), t.end());
    }

    print_csv(rows);
    return 0;
}