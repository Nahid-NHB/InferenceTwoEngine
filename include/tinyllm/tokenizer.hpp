// include/tinyllm/tokenizer.hpp
// -----------------------------------------------------------------------------
// Byte-level BPE tokenizer.
//
// We support a deliberately simple on-disk format that we can write ourselves
// from any existing tokenizer:
//
//   * vocab.json — JSON array of strings, index in array == token id.
//     The first 256 tokens MUST be byte tokens ("<0x00>", "<0x01>", …,
//     "<0xFF>") so we can map UTF-8 bytes to base tokens deterministically.
//     The token at index 0 is conventionally <unk>.
//   * merges.txt — plain text, one merge per line, in the order they should
//     be applied. Lower line number == higher priority (applied first).
//   * Optionally a JSON special-tokens file mapping strings to ids.
//
// Why this format?
//   - Trivially writable by hand.
//   - Trivially generated from any LLaMA/HuggingFace tokenizer by a small
//     helper script (we ship one in examples/).
//   - No dependency on protobuf / sentencepiece.
//
// Encoding (text -> ids):
//   1. UTF-8 encode the input.
//   2. Map each byte to its base byte-token (id 0..255).
//   3. While an adjacent pair has a merge rank in the loaded set: merge
//      the lowest-ranked pair.
//   4. Optionally prepend BOS / append EOS.
//
// Decoding (ids -> text):
//   1. For each id, find the underlying bytes (byte tokens map to a single
//      byte; everything else maps via id_to_token_).
//   2. Join as UTF-8.
//
// This is intentionally simple — not 100% byte-for-byte identical to any
// specific LLaMA release — but it is enough to round-trip text and to
// demonstrate the algorithm. Loading a real tokenizer.model (SentencePiece)
// or tokenizer.json (HuggingFace) is a Phase 3.5 extension.
// -----------------------------------------------------------------------------
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace tinyllm {

class BpeTokenizer {
public:
    struct Stats {
        std::size_t vocab_size = 0;
        std::size_t n_merges   = 0;
    };

    // Load from on-disk files.
    static BpeTokenizer load(std::string_view vocab_json_path,
                             std::string_view merges_txt_path);

    // Load from in-memory blobs (handy for tests).
    static BpeTokenizer load_from_memory(std::string_view vocab_json,
                                         std::string_view merges_txt);

    // Encode / decode.
    std::vector<int32_t> encode(std::string_view text,
                                bool add_bos = false,
                                bool add_eos = false) const;

    std::string decode(const std::vector<int32_t>& ids) const;

    // Lookup.
    std::optional<int32_t> token_to_id(std::string_view tok) const;
    const std::string&     id_to_token(int32_t id) const;

    // Sizes.
    std::size_t vocab_size() const noexcept { return id_to_token_.size(); }
    std::size_t n_merges()   const noexcept { return merge_rank_.size(); }

    // Special token IDs.
    int32_t bos_id() const noexcept { return bos_id_; }
    int32_t eos_id() const noexcept { return eos_id_; }
    int32_t unk_id() const noexcept { return 0; }

private:
    BpeTokenizer() = default;

    // Build merge_rank_ and the ordered merges_ list from a raw list of
    // "A B" pairs in priority order.
    void build_merge_ranks(const std::vector<std::pair<std::string, std::string>>& merges);

    // id <-> token tables.
    std::unordered_map<std::string, int32_t> token_to_id_;
    std::vector<std::string>                id_to_token_;

    // merge rank: lower rank (earlier in the merges file) == merge sooner.
    // We use the pair "A\x01B" as the key (so the separator doesn't appear
    // in any real token).
    std::unordered_map<std::string, int32_t>        merge_rank_;
    std::vector<std::pair<std::string, std::string>> merges_;

    int32_t bos_id_ = -1;
    int32_t eos_id_ = -1;
};

}  // namespace tinyllm