// src/tokenizer.cpp
// -----------------------------------------------------------------------------
// BPE tokenizer implementation.
//
// We keep the on-disk parser deliberately minimal:
//   - vocab.json: must be a flat JSON array of double-quoted strings. We
//     reject whitespace outside strings and handle \", \\, \/, \n, \t, \r,
//     \b, \f, \uXXXX.
//   - merges.txt: each non-empty line is "<a> <b>" (split on a single space).
//
// Encoding follows the canonical BPE loop: keep finding the lowest-ranked
// adjacent pair and merging it until no more pairs are in the merge table.
//
// Decoding:
//   - byte tokens (id 0..255) -> their single byte (inverse of byte->token).
//   - everything else        -> the token string as raw UTF-8 bytes.
// -----------------------------------------------------------------------------
#include "tinyllm/tokenizer.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace tinyllm {

namespace {

// -----------------------------------------------------------------------------
// Tiny JSON array-of-strings parser. Returns a vector of strings.
//
// We do NOT implement full JSON — only what vocab.json actually contains:
// a flat array of strings. This keeps us free of dependencies.
// -----------------------------------------------------------------------------
class VocabJsonParser {
public:
    explicit VocabJsonParser(std::string_view src) : src_(src) {}

    std::vector<std::string> parse() {
        skip_ws();
        expect('[');
        std::vector<std::string> out;
        while (true) {
            skip_ws();
            if (peek() == ']') { ++p_; break; }
            out.push_back(parse_string());
            skip_ws();
            if (peek() == ',') { ++p_; continue; }
            if (peek() == ']') { ++p_; break; }
            throw std::runtime_error("vocab.json: expected ',' or ']'");
        }
        skip_ws();
        if (p_ != src_.size()) {
            throw std::runtime_error("vocab.json: trailing content");
        }
        return out;
    }

private:
    char peek() const {
        if (p_ >= src_.size()) {
            throw std::runtime_error("vocab.json: unexpected end of input");
        }
        return src_[p_];
    }
    void expect(char c) {
        if (peek() != c) {
            throw std::runtime_error(std::string("vocab.json: expected '") + c + "'");
        }
        ++p_;
    }
    void skip_ws() {
        while (p_ < src_.size()) {
            char c = src_[p_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') ++p_;
            else break;
        }
    }
    std::string parse_string() {
        expect('"');
        std::string out;
        while (true) {
            if (p_ >= src_.size()) {
                throw std::runtime_error("vocab.json: unterminated string");
            }
            char c = src_[p_++];
            if (c == '"') break;
            if (c == '\\') {
                if (p_ >= src_.size()) {
                    throw std::runtime_error("vocab.json: bad escape");
                }
                char e = src_[p_++];
                switch (e) {
                    case '"': out += '"'; break;
                    case '\\': out += '\\'; break;
                    case '/': out += '/'; break;
                    case 'b': out += '\b'; break;
                    case 'f': out += '\f'; break;
                    case 'n': out += '\n'; break;
                    case 'r': out += '\r'; break;
                    case 't': out += '\t'; break;
                    case 'u': {
                        if (p_ + 4 > src_.size()) {
                            throw std::runtime_error("vocab.json: bad \\u");
                        }
                        unsigned cp = 0;
                        for (int i = 0; i < 4; ++i) {
                            char h = src_[p_++];
                            cp <<= 4;
                            if (h >= '0' && h <= '9') cp |= (h - '0');
                            else if (h >= 'a' && h <= 'f') cp |= (h - 'a' + 10);
                            else if (h >= 'A' && h <= 'F') cp |= (h - 'A' + 10);
                            else throw std::runtime_error("vocab.json: bad hex in \\u");
                        }
                        // Encode the code point as UTF-8.
                        if (cp < 0x80) {
                            out += static_cast<char>(cp);
                        } else if (cp < 0x800) {
                            out += static_cast<char>(0xC0 | (cp >> 6));
                            out += static_cast<char>(0x80 | (cp & 0x3F));
                        } else if (cp < 0x10000) {
                            out += static_cast<char>(0xE0 | (cp >> 12));
                            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                            out += static_cast<char>(0x80 | (cp & 0x3F));
                        } else {
                            out += static_cast<char>(0xF0 | (cp >> 18));
                            out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
                            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                            out += static_cast<char>(0x80 | (cp & 0x3F));
                        }
                        break;
                    }
                    default:
                        throw std::runtime_error("vocab.json: bad escape");
                }
            } else {
                out += c;
            }
        }
        return out;
    }

    std::string_view src_;
    std::size_t p_ = 0;
};

// -----------------------------------------------------------------------------
// Helpers for the byte-token convention.
// GPT-2 / LLaMA use "<0xNN>" as the printable form of byte NN.
// -----------------------------------------------------------------------------

// Inverse: extract the byte from a "<0xNN>" string. Returns -1 if not a
// byte token. We only use this when decoding, so we know id 0..255 are
// byte tokens by construction.
int parse_byte_token(std::string_view s) {
    if (s.size() != 6) return -1;
    if (s[0] != '<' || s[1] != '0' || s[2] != 'x' || s[5] != '>') return -1;
    auto hex = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    int hi = hex(s[3]);
    int lo = hex(s[4]);
    if (hi < 0 || lo < 0) return -1;
    return (hi << 4) | lo;
}

// Merges file line format: "A B" (one space separator). Empty / blank lines
// are skipped; lines starting with '#' are treated as comments.
std::vector<std::pair<std::string, std::string>> parse_merges(std::istream& in) {
    std::vector<std::pair<std::string, std::string>> out;
    std::string line;
    while (std::getline(in, line)) {
        // Trim trailing \r and surrounding whitespace.
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t')) {
            line.pop_back();
        }
        std::size_t start = 0;
        while (start < line.size() && (line[start] == ' ' || line[start] == '\t')) {
            ++start;
        }
        if (start == line.size()) continue;            // empty
        if (line[start] == '#') continue;             // comment
        auto sp = line.find(' ', start);
        if (sp == std::string::npos) {
            throw std::runtime_error("merges.txt: line has no space separator: '" + line + "'");
        }
        std::string a = line.substr(start, sp - start);
        std::string b = line.substr(sp + 1);
        out.emplace_back(std::move(a), std::move(b));
    }
    return out;
}

}  // namespace

// -----------------------------------------------------------------------------
// Lifecycle
// -----------------------------------------------------------------------------
BpeTokenizer BpeTokenizer::load(std::string_view vocab_json_path,
                                std::string_view merges_txt_path) {
    std::ifstream vf{std::string(vocab_json_path)};
    if (!vf) throw std::runtime_error("tokenizer: cannot open vocab.json: " + std::string(vocab_json_path));
    std::ostringstream vb;
    vb << vf.rdbuf();
    std::ifstream mf{std::string(merges_txt_path)};
    if (!mf) throw std::runtime_error("tokenizer: cannot open merges.txt: " + std::string(merges_txt_path));
    std::ostringstream mb;
    mb << mf.rdbuf();
    return load_from_memory(vb.str(), mb.str());
}

BpeTokenizer BpeTokenizer::load_from_memory(std::string_view vocab_json,
                                           std::string_view merges_txt) {
    BpeTokenizer t;

    // Parse vocab JSON.
    VocabJsonParser vp(vocab_json);
    t.id_to_token_ = vp.parse();

    if (t.id_to_token_.size() < 256) {
        throw std::runtime_error("vocab.json: must have at least 256 entries (the byte tokens)");
    }
    // Validate the first 256 entries follow the "<0xNN>" convention.
    for (int i = 0; i < 256; ++i) {
        int b = parse_byte_token(t.id_to_token_[i]);
        if (b != i) {
            throw std::runtime_error(
                "vocab.json: entry " + std::to_string(i) +
                " should be byte token for 0x" +
                std::to_string(i) + " but is '" + t.id_to_token_[i] + "'");
        }
    }

    // Build the lookup table.
    t.token_to_id_.reserve(t.id_to_token_.size());
    for (std::size_t i = 0; i < t.id_to_token_.size(); ++i) {
        t.token_to_id_[t.id_to_token_[i]] = static_cast<int32_t>(i);
    }

    // Common special tokens (best-effort, optional).
    if (auto it = t.token_to_id_.find("<bos>"); it != t.token_to_id_.end()) {
        t.bos_id_ = it->second;
    }
    if (auto it = t.token_to_id_.find("<eos>"); it != t.token_to_id_.end()) {
        t.eos_id_ = it->second;
    }

    // Parse merges.
    std::istringstream ms{std::string(merges_txt)};
    auto merges = parse_merges(ms);
    t.build_merge_ranks(merges);

    return t;
}

void BpeTokenizer::build_merge_ranks(const std::vector<std::pair<std::string, std::string>>& merges) {
    merges_ = merges;
    merge_rank_.clear();
    merge_rank_.reserve(merges.size() * 2);
    for (std::size_t i = 0; i < merges.size(); ++i) {
        // Use a non-printable separator that can't appear in any real token.
        std::string key;
        key.reserve(merges[i].first.size() + merges[i].second.size() + 1);
        key += merges[i].first;
        key += '\x01';
        key += merges[i].second;
        merge_rank_[std::move(key)] = static_cast<int32_t>(i);
    }
}

// -----------------------------------------------------------------------------
// Lookup
// -----------------------------------------------------------------------------
std::optional<int32_t> BpeTokenizer::token_to_id(std::string_view tok) const {
    auto it = token_to_id_.find(std::string(tok));
    if (it == token_to_id_.end()) return std::nullopt;
    return it->second;
}
const std::string& BpeTokenizer::id_to_token(int32_t id) const {
    static const std::string kEmpty;
    if (id < 0 || id >= static_cast<int32_t>(id_to_token_.size())) return kEmpty;
    return id_to_token_[id];
}

// -----------------------------------------------------------------------------
// BPE encode
// -----------------------------------------------------------------------------
std::vector<int32_t> BpeTokenizer::encode(std::string_view text,
                                          bool add_bos,
                                          bool add_eos) const {
    // 1. UTF-8 bytes -> base byte tokens.
    std::vector<int32_t> ids;
    ids.reserve(text.size() + 2);
    for (unsigned char c : text) {
        // The byte tokens are at positions 0..255 in our convention.
        ids.push_back(static_cast<int32_t>(c));
    }

    // 2. BPE merge loop. Represent the working word as a vector of strings.
    //    Use the merge ranks to repeatedly merge the lowest-ranked pair.
    //    We maintain a parallel vector of ranks-of-each-token to make pair
    //    ranking fast.
    //
    //    We always rebuild the working representation as a vector of
    //    strings (the *displayed* tokens) and ranks.
    if (ids.size() >= 2) {
        // Convert ids -> strings (using id_to_token_).
        std::vector<std::string> word;
        word.reserve(ids.size());
        for (int32_t id : ids) {
            word.push_back(id_to_token(id));   // for byte tokens this is "<0xNN>"
        }

        while (word.size() >= 2) {
            // Find the pair with the lowest rank.
            int32_t best_rank = std::numeric_limits<int32_t>::max();
            std::size_t best_idx = 0;
            bool found = false;
            for (std::size_t i = 0; i + 1 < word.size(); ++i) {
                std::string key;
                key.reserve(word[i].size() + word[i + 1].size() + 1);
                key += word[i];
                key += '\x01';
                key += word[i + 1];
                auto it = merge_rank_.find(key);
                if (it != merge_rank_.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_idx = i;
                    found = true;
                }
            }
            if (!found) break;

            // Merge at best_idx: word[best_idx] + word[best_idx+1] -> new symbol.
            // We look up the merged string by token_id matching (it should be in
            // the vocab as some token id; the canonical name is the literal
            // concatenation of the two sub-tokens).
            std::string merged = word[best_idx] + word[best_idx + 1];
            word[best_idx] = std::move(merged);
            word.erase(word.begin() + best_idx + 1);
        }

        // 3. Convert word -> ids via token_to_id_.
        ids.clear();
        ids.reserve(word.size());
        for (const std::string& tok : word) {
            auto it = token_to_id_.find(tok);
            if (it == token_to_id_.end()) {
                ids.push_back(unk_id());
            } else {
                ids.push_back(it->second);
            }
        }
    }

    // 4. Special tokens.
    if (add_bos && bos_id_ >= 0) ids.insert(ids.begin(), bos_id_);
    if (add_eos && eos_id_ >= 0) ids.push_back(eos_id_);

    return ids;
}

// -----------------------------------------------------------------------------
// BPE decode
// -----------------------------------------------------------------------------
std::string BpeTokenizer::decode(const std::vector<int32_t>& ids) const {
    std::string out;
    out.reserve(ids.size());
    for (int32_t id : ids) {
        if (id < 0 || id >= static_cast<int32_t>(id_to_token_.size())) continue;
        const std::string& tok = id_to_token_[id];
        if (id < 256) {
            // Byte token: emit the byte directly.
            out.push_back(static_cast<char>(id));
        } else {
            out += tok;
        }
    }
    return out;
}

}  // namespace tinyllm