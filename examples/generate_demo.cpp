// examples/generate_demo.cpp
// -----------------------------------------------------------------------------
// End-to-end demo of the inference stack.
//
// Walks the full pipeline:
//   1. Build a LlamaConfig from CLI flags (defaults to a tiny model that
//      finishes in <1s on a single thread).
//   2. Construct a LlamaModel with deterministic random weights.
//   3. Run generate() with a few sampling strategies (greedy, top-k,
//      top-p, hot sampling) and print the generated token sequence.
//   4. Run the fused Q4_0×F32 AVX2 kernel on a real-weight-shape matrix
//      and print the wall-time vs an equivalent F32 matmul.
//
// This is the closest thing we have to "run the model" without a real
// GGUF file (Phase 4's parser exists but a full GGUF → LlamaModel
// loader is not yet wired). All forward / cache / sampling kernels are
// exercised by this driver.
//
// Flags:
//   --hidden N       hidden dim (default 64)
//   --heads N        query heads (default 4)
//   --kv-heads N     KV heads; GQA if kv < heads (default 2)
//   --layers N       number of blocks (default 2)
//   --vocab N        vocab size (default 64)
//   --max-new N      tokens to generate (default 16)
//   --max-seq N      KV cache capacity (default 128)
//   --prompt T1,T2,...  comma-separated prompt token ids (default 0,1,2)
//   --seed N         PRNG seed for weights and sampling (default 7)
//   --q4-only        run only the quantized-matvec section, skip
//                    the model-generation section
// -----------------------------------------------------------------------------
#include "tinyllm/matmul.hpp"
#include "tinyllm/model.hpp"
#include "tinyllm/quantize.hpp"
#include "tinyllm/sampler.hpp"
#include "tinyllm/tensor.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace tinyllm;
using clk = std::chrono::high_resolution_clock;

struct Args {
    int64_t hidden       = 64;
    int64_t n_heads      = 4;
    int64_t n_kv_heads   = 2;     // GQA on by default
    int64_t n_layers     = 2;
    int64_t vocab        = 64;
    int64_t max_new      = 16;
    int64_t max_seq      = 128;
    int64_t seed         = 7;
    std::vector<int32_t> prompt = {0, 1, 2};
    bool q4_only = false;
};

static Args parse_args(int argc, char** argv) {
    Args a;
    auto take = [&](int& i, const char* what) -> const char* {
        if (i + 1 >= argc) {
            std::fprintf(stderr, "missing value after %s\n", what);
            std::exit(2);
        }
        ++i;
        return argv[i];
    };
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if      (k == "--hidden")  a.hidden  = std::atoll(take(i, k.c_str()));
        else if (k == "--heads")   a.n_heads = std::atoll(take(i, k.c_str()));
        else if (k == "--kv-heads")a.n_kv_heads = std::atoll(take(i, k.c_str()));
        else if (k == "--layers")  a.n_layers = std::atoll(take(i, k.c_str()));
        else if (k == "--vocab")   a.vocab   = std::atoll(take(i, k.c_str()));
        else if (k == "--max-new") a.max_new = std::atoll(take(i, k.c_str()));
        else if (k == "--max-seq") a.max_seq = std::atoll(take(i, k.c_str()));
        else if (k == "--seed")    a.seed    = std::atoll(take(i, k.c_str()));
        else if (k == "--prompt") {
            a.prompt.clear();
            const char* s = take(i, k.c_str());
            for (const char* p = s; *p; ) {
                char* end = nullptr;
                long v = std::strtol(p, &end, 10);
                if (end == p) break;
                a.prompt.push_back(static_cast<int32_t>(v));
                p = end;
                if (*p == ',') ++p;
            }
        }
        else if (k == "--q4-only") a.q4_only = true;
        else if (k == "--help" || k == "-h") {
            std::printf("Usage: %s [--hidden N] [--heads N] [--kv-heads N] "
                        "[--layers N] [--vocab N] [--max-new N] [--max-seq N] "
                        "[--seed N] [--prompt T1,T2,...] [--q4-only]\n",
                        argv[0]);
            std::exit(0);
        }
        else {
            std::fprintf(stderr, "unknown flag: %s\n", k.c_str());
            std::exit(2);
        }
    }
    return a;
}

// Deterministic weight filler (LCG so the run is reproducible).
struct Lcg {
    uint64_t s;
    explicit Lcg(uint64_t seed) : s(seed ? seed : 1) {}
    float next_unit() {
        // Output in [-0.05, +0.05] — small magnitude keeps logits sane.
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        uint32_t bits = static_cast<uint32_t>((s >> 11) & ((1ULL << 21) - 1));
        return (static_cast<float>(bits) / static_cast<float>(1ULL << 21)) * 0.1f - 0.05f;
    }
};

static LlamaModel build_model(const LlamaConfig& cfg, uint64_t seed) {
    Lcg rng(seed);
    auto fill = [&](Tensor& t, float scale) {
        for (int64_t i = 0; i < t.numel(); ++i) {
            t.at_flat(i) = rng.next_unit() * scale;
        }
    };
    LlamaModelWeights w;
    w.W_embed   = Tensor({cfg.vocab_size, cfg.hidden}, DType::Float32);
    fill(w.W_embed, 1.0f);
    w.final_norm = Tensor({cfg.hidden}, DType::Float32);
    for (int64_t i = 0; i < cfg.hidden; ++i) w.final_norm.at_flat(i) = 1.0f;
    w.W_output  = Tensor({cfg.hidden, cfg.vocab_size}, DType::Float32);
    fill(w.W_output, 1.0f);

    w.blocks.resize(static_cast<std::size_t>(cfg.n_layers));
    int64_t H  = cfg.hidden;
    int64_t I  = cfg.intermediate;
    int64_t Hd = cfg.head_dim;
    int64_t KH = cfg.n_kv_heads;
    int64_t QH = cfg.n_heads * Hd;
    for (auto& b : w.blocks) {
        b.weights.attn_norm = Tensor({H}, DType::Float32);
        b.weights.mlp_norm  = Tensor({H}, DType::Float32);
        for (int64_t i = 0; i < H; ++i) {
            b.weights.attn_norm.at_flat(i) = 1.0f;
            b.weights.mlp_norm.at_flat(i)  = 1.0f;
        }
        b.weights.attn.Wq = Tensor({H, QH}, DType::Float32);
        b.weights.attn.Wk = Tensor({H, KH * Hd}, DType::Float32);
        b.weights.attn.Wv = Tensor({H, KH * Hd}, DType::Float32);
        b.weights.attn.Wo = Tensor({QH, H}, DType::Float32);
        b.weights.mlp.W_gate = Tensor({H, I}, DType::Float32);
        b.weights.mlp.W_up   = Tensor({H, I}, DType::Float32);
        b.weights.mlp.W_down = Tensor({I, H}, DType::Float32);
        for (auto* t : {&b.weights.attn.Wq, &b.weights.attn.Wk,
                        &b.weights.attn.Wv, &b.weights.attn.Wo,
                        &b.weights.mlp.W_gate, &b.weights.mlp.W_up,
                        &b.weights.mlp.W_down}) {
            fill(*t, 1.0f);
        }
    }
    return make_model(std::move(w), cfg);
}

static void print_token_seq(const std::vector<int32_t>& ids,
                            const char* label) {
    std::printf("%-26s [", label);
    for (std::size_t i = 0; i < ids.size(); ++i) {
        std::printf("%s%d", i ? ", " : "", ids[i]);
    }
    std::printf("]\n");
}

// ---------------------------------------------------------------------------
// Q4_0 fused kernel demonstration: pack a real-shape MLP weight to Q4_0,
// time the AVX2 fused dot product vs an equivalent F32 matvec.
// ---------------------------------------------------------------------------
static void run_q4_demo(const Args& a) {
    // Use the MLP weight shape from the model we just built: hidden ×
    // intermediate. That makes this a representative workload rather
    // than a toy size.
    int64_t M = a.hidden;                  // = output dim (hidden)
    int64_t K = 4 * a.hidden;              // typical intermediate = 4·hidden
    std::printf("\n=== Q4_0 fused matvec (M=%ld, K=%ld) ===\n", M, K);

    Lcg rng(static_cast<uint64_t>(a.seed) * 31ULL + 1);
    std::vector<float> w(static_cast<std::size_t>(M * K));
    for (auto& v : w) v = rng.next_unit();
    std::vector<float> x(static_cast<std::size_t>(K));
    for (auto& v : x) v = rng.next_unit();

    auto packed = quantize_q4_0(w.data(), M * K);
    std::vector<float> y_q(static_cast<std::size_t>(M));
    std::vector<float> y_f(static_cast<std::size_t>(M));

    // Warmup.
    matmul_q4_0_f32(packed.data(), M, K, x.data(), y_q.data());

    // Time Q4_0 fused.
    auto t0 = clk::now();
    const int reps = 200;
    for (int i = 0; i < reps; ++i) {
        matmul_q4_0_f32(packed.data(), M, K, x.data(), y_q.data());
    }
    auto t1 = clk::now();
    double q_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;

    // Reference: Tensor matmul on a column vector.
    Tensor W({M, K}, DType::Float32, w.data(), w.size() * sizeof(float));
    Tensor X({K, 1}, DType::Float32, x.data(), x.size() * sizeof(float));
    auto warm = ops::matmul(W, X, ops::MatmulVariant::Avx2);
    (void)warm;
    auto t2 = clk::now();
    for (int i = 0; i < reps; ++i) {
        auto r = ops::matmul(W, X, ops::MatmulVariant::Avx2);
        // Copy out for comparison.
        for (int64_t j = 0; j < M; ++j) y_f[j] = r.at_flat(j);
    }
    auto t3 = clk::now();
    double f_ms = std::chrono::duration<double, std::milli>(t3 - t2).count() / reps;

    std::printf("  Q4_0 fused : %7.3f ms  (%.1f ms total, %d reps)\n",
                q_ms, q_ms * reps, reps);
    std::printf("  F32 matmul : %7.3f ms  (auto variant: %s)\n",
                f_ms,
                std::string(ops::variant_name(ops::last_picked_variant())).c_str());
    std::printf("  speedup    : %.2fx\n", f_ms / q_ms);

    // Sanity: max abs diff between Q4_0 and F32. With small weights and
    // 16-level Q4_0 the relative error is <10% per element, dot-product
    // error scales with sqrt(K).
    double max_abs = 0.0;
    for (int64_t j = 0; j < M; ++j) {
        max_abs = std::max(max_abs,
            std::fabs(static_cast<double>(y_q[j]) - static_cast<double>(y_f[j])));
    }
    std::printf("  max |Q4_0 - F32| = %.4f  (sanity)\n", max_abs);
}

int main(int argc, char** argv) {
    Args a = parse_args(argc, argv);

    std::printf("tinyllm generate_demo\n");
    std::printf("  hidden=%ld  heads=%ld  kv_heads=%ld  layers=%ld  vocab=%ld\n",
                a.hidden, a.n_heads, a.n_kv_heads, a.n_layers, a.vocab);
    std::printf("  max_new=%ld  max_seq=%ld  seed=%ld\n",
                a.max_new, a.max_seq, a.seed);
    std::printf("  AVX2=%s  threads=%d\n",
                ops::have_avx2() ? "yes" : "no",
                ops::hardware_threads());

    LlamaConfig cfg;
    cfg.vocab_size   = a.vocab;
    cfg.hidden       = a.hidden;
    cfg.intermediate = 4 * a.hidden;
    cfg.n_heads      = a.n_heads;
    cfg.n_kv_heads   = a.n_kv_heads;
    cfg.head_dim     = a.hidden / a.n_heads;
    cfg.n_layers     = a.n_layers;
    cfg.max_seq_len  = a.max_seq;

    if (cfg.head_dim * a.n_heads != a.hidden) {
        std::fprintf(stderr,
                     "head_dim * n_heads (%ld * %ld = %ld) must equal hidden (%ld)\n",
                     cfg.head_dim, a.n_heads, cfg.head_dim * a.n_heads, a.hidden);
        return 2;
    }

    if (cfg.n_kv_heads > cfg.n_heads) {
        std::fprintf(stderr,
                     "n_kv_heads (%ld) cannot exceed n_heads (%ld)\n",
                     cfg.n_kv_heads, cfg.n_heads);
        return 2;
    }

    for (int32_t t : a.prompt) {
        if (t < 0 || t >= cfg.vocab_size) {
            std::fprintf(stderr, "prompt token %d out of range [0, %ld)\n",
                         t, cfg.vocab_size);
            return 2;
        }
    }

    if (!a.q4_only) {
        LlamaModel model = build_model(cfg, static_cast<uint64_t>(a.seed));

        std::printf("\n=== Generation with random-init weights ===\n");
        std::printf("(the model has not been trained; this is a sanity check\n"
                    " that the full forward-cache-sample loop runs end-to-end)\n\n");

        print_token_seq(a.prompt, "prompt");

        // 1. Greedy (deterministic).
        SamplerConfig greedy;
        greedy.temperature = 0.0f;
        auto t0 = clk::now();
        auto out_greedy = generate(model, a.prompt, a.max_new, greedy,
                                   /*eos=*/std::nullopt,
                                   /*seed=*/static_cast<uint64_t>(a.seed));
        auto t1 = clk::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        print_token_seq(out_greedy, "greedy");
        std::printf("  %ld tokens in %.1f ms (%.2f ms/tok)\n\n",
                    out_greedy.size(), ms,
                    out_greedy.empty() ? 0.0 : ms / out_greedy.size());

        // 2. Sampling with temperature.
        SamplerConfig sampled;
        sampled.temperature = 0.8f;
        auto out_s = generate(model, a.prompt, a.max_new, sampled,
                              /*eos=*/std::nullopt,
                              /*seed=*/static_cast<uint64_t>(a.seed * 2 + 1));
        print_token_seq(out_s, "T=0.8 sample");

        // 3. Top-k.
        SamplerConfig topk;
        topk.temperature = 1.0f;
        topk.top_k = 4;
        auto out_k = generate(model, a.prompt, a.max_new, topk,
                              /*eos=*/std::nullopt,
                              /*seed=*/static_cast<uint64_t>(a.seed * 3 + 7));
        print_token_seq(out_k, "T=1.0 top_k=4");

        // 4. Top-p.
        SamplerConfig topp;
        topp.temperature = 1.0f;
        topp.top_p = 0.9f;
        auto out_p = generate(model, a.prompt, a.max_new, topp,
                              /*eos=*/std::nullopt,
                              /*seed=*/static_cast<uint64_t>(a.seed * 5 + 11));
        print_token_seq(out_p, "T=1.0 top_p=0.9");

        // 5. Deterministic with same seed: greedy twice must match.
        auto out_g2 = generate(model, a.prompt, a.max_new, greedy,
                               /*eos=*/std::nullopt,
                               /*seed=*/static_cast<uint64_t>(a.seed));
        bool same = (out_greedy == out_g2);
        std::printf("\n  deterministic-check (greedy twice): %s\n",
                     same ? "MATCH" : "MISMATCH");
        if (!same) {
            print_token_seq(out_g2, "  second run");
        }
    }

    run_q4_demo(a);

    std::printf("\nDone.\n");
    return 0;
}