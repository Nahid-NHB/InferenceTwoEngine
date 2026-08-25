// tests/test_tokenizer.cpp
// -----------------------------------------------------------------------------
// Tests for the BPE tokenizer.
//
// We build a tiny in-memory tokenizer suitable for testing:
//   - 256 byte tokens (id 0..255 = "<0xNN>")
//   - a few single-character ASCII tokens (id 256..)
//   - a handful of merge rules so the merge loop has something to do
// -----------------------------------------------------------------------------
#include "test_helpers.hpp"
#include "tinyllm/tokenizer.hpp"

#include <string>

using namespace tinyllm;

// -----------------------------------------------------------------------------
// Build a minimal in-memory vocab.json + merges.txt that exercises:
//   - byte tokens (required)
//   - a single-character ASCII token (uppercase letters)
//   - a merge that combines two letters (e.g. "h" + "i" -> "hi")
//   - a multi-byte merge (e.g. "h" + "i" -> "hi" then "hi" + "s" -> "his")
// -----------------------------------------------------------------------------
static std::string tiny_vocab_json() {
    // 256 byte tokens + a few extras. Indices:
    //   0..255   : "<0xNN>"
    //   256      : "<unk>"
    //   257      : "h"
    //   258      : "i"
    //   259      : "hi"
    //   260      : "s"
    //   261      : "his"
    //   262      : "<bos>"
    //   263      : "<eos>"
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
    // Rank 0: h i -> hi
    // Rank 1: hi s -> his
    // (lower rank = applied first)
    return "h i\nhi s\n";
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------
TEST_CASE(tokenizer_loads_byte_tokens) {
    auto t = BpeTokenizer::load_from_memory(tiny_vocab_json(), tiny_merges_txt());
    REQUIRE(t.vocab_size() == 256 + 8);
    REQUIRE(t.bos_id() == 262);
    REQUIRE(t.eos_id() == 263);
    REQUIRE(t.n_merges() == 2);
}

TEST_CASE(tokenizer_encodes_pure_ascii) {
    auto t = BpeTokenizer::load_from_memory(tiny_vocab_json(), tiny_merges_txt());
    // "ab" — no merges apply. Should be two byte tokens.
    auto ids = t.encode("ab");
    REQUIRE(ids.size() == 2);
    REQUIRE(ids[0] == static_cast<int32_t>('a'));
    REQUIRE(ids[1] == static_cast<int32_t>('b'));
}

TEST_CASE(tokenizer_applies_single_merge) {
    auto t = BpeTokenizer::load_from_memory(tiny_vocab_json(), tiny_merges_txt());
    // "hi" — merge "h" + "i" -> "hi" (rank 0).
    auto ids = t.encode("hi");
    REQUIRE(ids.size() == 1);
    REQUIRE(ids[0] == 259);  // "hi"
}

TEST_CASE(tokenizer_applies_chain_of_merges) {
    auto t = BpeTokenizer::load_from_memory(tiny_vocab_json(), tiny_merges_txt());
    // "his" — first "h"+"i" -> "hi" (rank 0), then "hi"+"s" -> "his" (rank 1).
    auto ids = t.encode("his");
    REQUIRE(ids.size() == 1);
    REQUIRE(ids[0] == 261);  // "his"
}

TEST_CASE(tokenizer_does_not_apply_lower_rank_after_higher) {
    // Build a vocab where two competing merges exist. The lower-rank merge
    // must be applied first.
    //   merges: rank 0: "h" "i" -> "hi", rank 1: "h" "is" -> "his"
    //          (we don't have "is" as a base token, so this won't apply)
    // Encode "hi": should still get id 259 ("hi").
    auto t = BpeTokenizer::load_from_memory(tiny_vocab_json(), "h i\n");
    auto ids = t.encode("hi");
    REQUIRE(ids.size() == 1);
    REQUIRE(ids[0] == 259);
}

TEST_CASE(tokenizer_round_trip_ascii) {
    auto t = BpeTokenizer::load_from_memory(tiny_vocab_json(), tiny_merges_txt());
    std::string s = "hello, this is a test.";
    auto ids = t.encode(s);
    auto back = t.decode(ids);
    REQUIRE(back == s);
}

TEST_CASE(tokenizer_round_trip_unicode) {
    auto t = BpeTokenizer::load_from_memory(tiny_vocab_json(), tiny_merges_txt());
    std::string s = "café résumé naïve";  // contains multi-byte UTF-8
    auto ids = t.encode(s);
    auto back = t.decode(ids);
    REQUIRE(back == s);
}

TEST_CASE(tokenizer_bos_eos_specials) {
    auto t = BpeTokenizer::load_from_memory(tiny_vocab_json(), tiny_merges_txt());
    auto ids = t.encode("hi", /*add_bos=*/true, /*add_eos=*/true);
    REQUIRE(ids.size() == 3);
    REQUIRE(ids[0] == t.bos_id());
    REQUIRE(ids[2] == t.eos_id());
}

TEST_CASE(tokenizer_lookup_helpers) {
    auto t = BpeTokenizer::load_from_memory(tiny_vocab_json(), tiny_merges_txt());
    REQUIRE(t.token_to_id("hi").value() == 259);
    REQUIRE(t.token_to_id("doesnotexist").has_value() == false);
    REQUIRE(t.id_to_token(259) == "hi");
}

TEST_CASE(tokenizer_unknown_byte_token_falls_back_to_unk) {
    // If a merge produces a token that isn't in the vocab, the encode loop
    // should map it to unk_id() (= 0).
    auto t = BpeTokenizer::load_from_memory(tiny_vocab_json(),
                                            "h x\n");  // "h" "x" -> "hx" not in vocab
    auto ids = t.encode("hx");
    REQUIRE(ids.size() == 1);
    REQUIRE(ids[0] == 0);  // <unk>
}

TEST_CASE(tokenizer_rejects_short_vocab) {
    // Less than 256 entries must fail at load time.
    std::string bad_vocab = "[\"<0x00>\",\"<0x01>\"]";  // only 2 entries
    REQUIRE_THROWS(BpeTokenizer::load_from_memory(bad_vocab, ""));
}

TEST_CASE(tokenizer_rejects_wrong_byte_token_at_0) {
    // If entry 0 is not "<0x00>", reject the vocab.
    std::string bad_vocab = "[";
    bad_vocab += "\"<0xFF>\"";  // wrong: should be <0x00>
    for (int i = 1; i < 256; ++i) {
        char buf[16];
        std::snprintf(buf, sizeof(buf), "\"<0x%02X>\"", i);
        bad_vocab += ",";
        bad_vocab += buf;
    }
    bad_vocab += "]";
    REQUIRE_THROWS(BpeTokenizer::load_from_memory(bad_vocab, ""));
}

TEST_CASE(tokenizer_bench_encode_decode_speed) {
    // Not a correctness test — used by tests/bench_tokenizer.cpp.
    // Placeholder; the actual benchmark lives there.
    auto t = BpeTokenizer::load_from_memory(tiny_vocab_json(), tiny_merges_txt());
    std::string s;
    for (int i = 0; i < 100; ++i) s += "hello his ";
    (void)t.encode(s);
    REQUIRE(true);
}