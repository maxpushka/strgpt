#pragma once

#include <memory>
#include <regex>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#ifdef UNIT_TEST
#include <gtest/gtest_prod.h>
#endif

namespace tokenizer {
// hash_pair_wstring is used in BPERanks to make a pair of wstrings
// hashable, so the pair can be used as the key to unordered_map.
struct hash_pair_wstring {
  size_t operator()(const std::pair<std::wstring, std::wstring> &p) const {
    auto hash1 = std::hash<std::wstring>{}(p.first);
    auto hash2 = std::hash<std::wstring>{}(p.second);
    // If hash1 == hash2, their XOR is zero.
    return (hash1 != hash2) ? hash1 ^ hash2 : hash1;
  }
};

// BPERanks maps each merge rule, which is a pair of wstrings, to its
// rank.  This mapping allows quick lookup for the optimal merge rule.
using BPERanks = std::unordered_map<std::pair<std::wstring, std::wstring>, int,
                                    hash_pair_wstring>;

class BPE final {
 private:
  BPERanks bpe_ranks;
  std::regex re;
  std::unordered_map<std::string, int> t2i;  // token to id
  std::unordered_map<int, std::string> i2t;  // id to token
  std::unordered_map<uint8_t, wchar_t> b2u;
  std::unordered_map<wchar_t, uint8_t> u2b;

 public:
  BPE(std::istream &config_file, std::regex re);

  [[nodiscard]] std::vector<int> encode(const std::string &text) const;

  [[nodiscard]] std::string decode(const std::vector<int> &ids) const;

 private:

  void load_vocab(const nlohmann::json &ins);

  void load_merge_rules(const nlohmann::json &ins);

  void load_bytes_to_unicode();

  [[nodiscard]] static std::wstring utf8_to_wstring(const std::string &str);

  [[nodiscard]] static std::string utf8(const wchar_t c);

  [[nodiscard]] static std::string wstring_to_utf8(const std::wstring &str);

  [[nodiscard]] std::vector<std::string> tokenize(const std::string &text) const;

  void _tokenize(const std::string &text, std::vector<std::string> &result) const;

  // Given a token as a UTF8 string, encode each byte into an wchar_t
  [[nodiscard]] std::wstring byte_encode_token(const std::string &token) const;

  [[nodiscard]] std::vector<std::wstring> bpe(const std::wstring &token) const;

  static std::vector<std::pair<std::wstring, std::wstring>> get_pairs(const std::wstring &word);

#ifdef UNIT_TEST
  FRIEND_TEST(TokenizerBPE, RegexCompilation);
  FRIEND_TEST(TokenizerBPE, BytesToUnicodeConversion);
  FRIEND_TEST(TokenizerBPE, ByteEncodeToken);
  FRIEND_TEST(TokenizerBPE, LoadVocab);
  FRIEND_TEST(TokenizerBPE, LoadMergeRules);
  FRIEND_TEST(TokenizerBPE, GetPairs);
  FRIEND_TEST(TokenizerBPE, BPEAlgorithm);
  FRIEND_TEST(TokenizerBPE, Tokenize);
  FRIEND_TEST(TokenizerBPE, EncodeDecode);
#endif
};
}  // namespace tokenizer
