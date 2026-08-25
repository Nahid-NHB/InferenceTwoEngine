// benchmarks/bench_tokenizer.cpp
// -----------------------------------------------------------------------------
// BPE tokenizer encode/decode speed.
//
// For now we use the same in-memory toy vocab as the tests. With a real
// tokenizer file (~30 k merges) we'll be able to do meaningful measurements.
// -----------------------------------------------------------------------------
#include "tinyllm/tokenizer.hpp"

#include <chrono>
#include <cstdio>
#include <random>
#include <string>

using clk = std::chrono::high_resolution_clock;
using namespace tinyllm;

static std::string tiny_vocab_json() {
    std::string s = "[";
    for (int i = 0; i < 256; ++i) {
        char buf[16];
        std::snprintf(buf, sizeof(buf), "\"<0x%02X>\"", i);
        s += buf;
        if (i < 255) s += ",";
    }
    s += ",\"<unk>\",\"h\",\"i\",\"hi\",\"s\",\"his\",\"<bos>\",\"<eos>\"]";
    return s;
}
static std::string tiny_merges_txt() {
    return "h i\nhi s\n";
}

int main() {
    auto t = BpeTokenizer::load_from_memory(tiny_vocab_json(), tiny_merges_txt());
    std::printf("vocab_size = %zu, merges = %zu\n", t.vocab_size(), t.n_merges());

    // Generate a string with a mix of letters and spaces.
    std::string text;
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> letter(0, 25);
    for (int i = 0; i < 10'000; ++i) {
        text += 'a' + letter(rng);
        if (i % 100 == 0) text += ' ';
    }
    std::printf("input bytes = %zu\n", text.size());

    // Warmup
    auto warm = t.encode(text);
    (void)warm;

    auto t0 = clk::now();
    int iters = 100;
    std::vector<int32_t> last;
    for (int i = 0; i < iters; ++i) {
        last = t.encode(text);
    }
    auto t1 = clk::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("encode: %.3f ms/iter, %zu ids, %.2f bytes/ms\n",
                ms / iters, last.size(),
                static_cast<double>(text.size()) / (ms / iters));

    auto back = t.decode(last);
    std::printf("round-trip ok = %s\n", back == text ? "yes" : "NO");
    return 0;
}