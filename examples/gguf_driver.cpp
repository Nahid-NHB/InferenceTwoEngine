// examples/gguf_driver.cpp
// -----------------------------------------------------------------------------
// End-to-end driver that loads a real Llama-family GGUF and runs generate.
//
//   ./bin/gguf_driver --gguf tinyllama-1.1b-q4_0.gguf \
//                     --prompt 1,2,3 --max-new 32 --temperature 0
//
// Token IDs only — no tokenizer shim. The user is responsible for
// converting text -> token ids (e.g. via the SentencePiece / BPE tool
// that produced the GGUF in the first place) and for mapping the
// printed ids back to text.
//
// The CLI prints:
//   - the loaded LlamaConfig
//   - the estimated F32 weight memory (with a >4 GiB soft warning if it
//     would exceed the threshold)
//   - the prompt token ids
//   - the generated token ids
//   - a hex view of each generated id (sanity / debugging)
//   - a tok/s wall-time number
//
// Flags:
//   --gguf PATH        path to a GGUF v3 file (required)
//   --prompt T1,T2,...  comma-separated prompt token ids (default 1,2,3)
//   --max-new N        tokens to generate (default 16)
//   --seed N           PRNG seed for sampling (default 7)
//   --temperature F    sampling temperature; <=0 -> greedy (default 0)
//   --top-k N          keep only top-k logits (default 0 = disabled)
//   --top-p F          nucleus sampling threshold (default 1.0 = disabled)
//   --help, -h         show usage
// -----------------------------------------------------------------------------
#include "tinyllm/gguf.hpp"
#include "tinyllm/llama_loader.hpp"
#include "tinyllm/model.hpp"
#include "tinyllm/sampler.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

using namespace tinyllm;
using clk = std::chrono::high_resolution_clock;

struct Args {
    std::string gguf_path;
    std::vector<int32_t> prompt = {1, 2, 3};
    int64_t max_new = 16;
    uint64_t seed = 7;
    float temperature = 0.0f;
    int top_k = 0;
    float top_p = 1.0f;
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
        if      (k == "--gguf")        a.gguf_path = take(i, k.c_str());
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
        else if (k == "--max-new")     a.max_new    = std::atoll(take(i, k.c_str()));
        else if (k == "--seed")        a.seed       = static_cast<uint64_t>(std::atoll(take(i, k.c_str())));
        else if (k == "--temperature") a.temperature = std::atof(take(i, k.c_str()));
        else if (k == "--top-k")       a.top_k       = std::atoi(take(i, k.c_str()));
        else if (k == "--top-p")       a.top_p       = std::atof(take(i, k.c_str()));
        else if (k == "--help" || k == "-h") {
            std::printf(
                "Usage: %s --gguf PATH [--prompt T1,T2,...] [--max-new N]\n"
                "          [--seed N] [--temperature F] [--top-k N] [--top-p F]\n",
                argv[0]);
            std::exit(0);
        }
        else {
            std::fprintf(stderr, "unknown flag: %s\n", k.c_str());
            std::exit(2);
        }
    }
    if (a.gguf_path.empty()) {
        std::fprintf(stderr, "error: --gguf PATH is required (use --help)\n");
        std::exit(2);
    }
    return a;
}

static void print_id_seq(const std::vector<int32_t>& ids, const char* label) {
    std::printf("%-12s [", label);
    for (std::size_t i = 0; i < ids.size(); ++i) {
        std::printf("%s%d", i ? ", " : "", ids[i]);
    }
    std::printf("]\n");
}

static void print_id_hex(const std::vector<int32_t>& ids, const char* label) {
    std::printf("%-12s [", label);
    for (std::size_t i = 0; i < ids.size(); ++i) {
        std::printf("%s0x%08x", i ? ", " : "", static_cast<uint32_t>(ids[i]));
    }
    std::printf("]\n");
}

int main(int argc, char** argv) {
    Args a = parse_args(argc, argv);

    std::printf("tinyllm gguf_driver\n");
    std::printf("  gguf         : %s\n", a.gguf_path.c_str());
    std::printf("  prompt       : %zu token(s)\n", a.prompt.size());
    std::printf("  max_new      : %ld\n", a.max_new);
    std::printf("  seed         : %lu\n", static_cast<unsigned long>(a.seed));
    std::printf("  temperature  : %.3f%s\n", a.temperature,
                a.temperature <= 0.0f ? " (greedy)" : "");
    if (a.top_k > 0) std::printf("  top_k        : %d\n", a.top_k);
    if (a.top_p < 1.0f) std::printf("  top_p        : %.3f\n", a.top_p);

    // ----- Load -----
    std::ostringstream diag;
    auto loaded = load_llama_from_gguf(a.gguf_path, diag);
    std::fputs(diag.str().c_str(), stdout);
    if (!loaded.has_value()) {
        std::fprintf(stderr, "error: load failed (see diagnostic above)\n");
        return 1;
    }

    auto& cfg = loaded->cfg;
    std::printf("\nloaded config:\n");
    std::printf("  vocab_size        : %ld\n", cfg.vocab_size);
    std::printf("  hidden            : %ld\n", cfg.hidden);
    std::printf("  intermediate      : %ld\n", cfg.intermediate);
    std::printf("  n_heads           : %ld\n", cfg.n_heads);
    std::printf("  n_kv_heads        : %ld\n", cfg.n_kv_heads);
    std::printf("  head_dim          : %ld\n", cfg.head_dim);
    std::printf("  n_layers          : %ld\n", cfg.n_layers);
    std::printf("  max_seq_len       : %ld\n", cfg.max_seq_len);
    std::printf("  rms_norm_eps      : %g\n",  cfg.rms_norm_eps);
    std::printf("  theta_base        : %g\n",  cfg.theta_base);

    // Down-stream warning if the F32-dequantized weights blow past 4 GiB.
    // The loader emits its own warning into diag; we also surface the
    // number here so the driver output is self-contained.
    try {
        GgufFile gf = GgufFile::open(a.gguf_path);
        std::size_t bytes = estimate_llama_f32_bytes(gf);
        std::printf("  F32 weight memory : %.1f MiB (%s)\n",
                    static_cast<double>(bytes) / (1024.0 * 1024.0),
                    bytes > kLlamaLoadSoftWarnBytes ? "OVER 4 GiB THRESHOLD" : "ok");
    } catch (...) {
        // Already-loaded weights are fine; the estimate is informational.
    }

    // Validate prompt range.
    for (int32_t t : a.prompt) {
        if (t < 0 || t >= cfg.vocab_size) {
            std::fprintf(stderr,
                         "error: prompt token %d out of range [0, %ld)\n",
                         t, cfg.vocab_size);
            return 2;
        }
    }
    if (a.prompt.empty()) {
        std::fprintf(stderr, "error: prompt is empty\n");
        return 2;
    }
    if (a.max_new <= 0) {
        std::fprintf(stderr, "error: --max-new must be > 0\n");
        return 2;
    }
    if (cfg.max_seq_len < static_cast<int64_t>(a.prompt.size()) + a.max_new) {
        std::fprintf(stderr,
                     "error: max_seq_len (%ld) < prompt (%zu) + max_new (%ld)\n",
                     cfg.max_seq_len, a.prompt.size(), a.max_new);
        return 2;
    }

    // ----- Build model + generate -----
    LlamaModel model = make_model(std::move(loaded->weights), cfg);

    SamplerConfig sc;
    sc.temperature = a.temperature;
    sc.top_k       = a.top_k;
    sc.top_p       = a.top_p;

    print_id_seq(a.prompt, "prompt");
    auto t0 = clk::now();
    auto out = generate(model, a.prompt,
                        static_cast<int>(a.max_new), sc,
                        /*eos=*/std::nullopt,
                        /*seed=*/a.seed);
    auto t1 = clk::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    print_id_seq(out, "generated");
    print_id_hex(out, "hex");
    std::printf("  %zu tokens in %.1f ms (%.2f ms/tok, %.1f tok/s)\n",
                out.size(), ms,
                out.empty() ? 0.0 : ms / out.size(),
                ms > 0 ? (out.size() * 1000.0 / ms) : 0.0);

    // Determinism sanity: greedy + same seed should reproduce.
    if (a.temperature <= 0.0f) {
        SamplerConfig greedy; greedy.temperature = 0.0f;
        auto out2 = generate(model, a.prompt,
                             static_cast<int>(a.max_new), greedy,
                             std::nullopt, a.seed);
        if (out == out2) {
            std::printf("  deterministic-check (greedy twice): MATCH\n");
        } else {
            std::printf("  deterministic-check (greedy twice): MISMATCH\n");
        }
    }
    return 0;
}