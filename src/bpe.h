#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include <re2/re2.h>
#include <re2/stringpiece.h>

#ifdef UNIT_TEST
#include <gtest/gtest_prod.h>
#endif

namespace bpe {
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
  std::unique_ptr<RE2> re;
  std::unordered_map<std::string, int> t2i;  // token to id
  std::unordered_map<int, std::string> i2t;  // id to token

 public:
  BPE(std::fstream &vocab_file, std::fstream &merges_file, std::unique_ptr<RE2> re);

  void encode(const std::string &text,
              std::unordered_map<uint8_t, wchar_t> &b2u,
              std::vector<int> *ids) const;

  std::string decode(const std::vector<int> &ids,
                     const std::unordered_map<wchar_t, uint8_t> &u2b) const;

 private:
  void load_vocab(std::istream &ins);

  void load_merge_rules(std::istream &ins);

  std::wstring utf8_to_wstring(const std::string &str) const;

  std::string wstring_to_utf8(const std::wstring &str) const;

  std::string utf8(wchar_t c) const;

  void tokenize(const std::string &text,
                const std::unordered_map<uint8_t, wchar_t> &b2u,
                std::vector<std::string> *result) const;

  void _tokenize(const std::string &text,
                 const std::unordered_map<uint8_t, wchar_t> &b2u,
                 std::vector<std::string> *result) const;

  // Given a token as a UTF8 string, encode each byte into an wchar_t
  void byte_encode_token(const std::string &token,
                         const std::unordered_map<uint8_t, wchar_t> &b2u,
                         std::wstring *result) const;

  void bpe(const std::wstring &token, std::vector<std::wstring> *result) const;

  static void get_pairs(const std::wstring &word,
                        std::vector<std::pair<std::wstring, std::wstring>> *pairs);

  static void bytes_to_unicode(std::unordered_map<uint8_t, wchar_t> *b2u,
                               std::unordered_map<wchar_t, uint8_t> *u2b);

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
}  // namespace bpe
