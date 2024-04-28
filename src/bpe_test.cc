#include "bpe.h"
#include <gtest/gtest.h>
#include <fstream>
#include <memory>

namespace bpe {
class TokenizerBPE : public testing::Test {
 protected:
  std::unique_ptr<bpe::BPE> bpe_instance;

  void SetUp() override {
    const char *assets_root = std::getenv("ASSETS_ROOT");
    ASSERT_TRUE(assets_root != nullptr);

    std::stringstream vocab_path, merges_path;
    vocab_path << assets_root << "/vocab.txt";
    merges_path << assets_root << "/merges.txt";

    std::fstream vocab_file(vocab_path.str(), std::ios::in);
    std::fstream merges_file(merges_path.str(), std::ios::in);

    auto re = std::make_unique<RE2>(
        "('s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| "
        "?[^\\s\\p{L}\\p{N}]+|\\s+\\(?!\\S\\)|\\s+)");

    this->bpe_instance = std::make_unique<bpe::BPE>(vocab_file, merges_file, std::move(re));
  }

  void TearDown() override {}
};

TEST_F(TokenizerBPE, RegexCompilation) {
  EXPECT_TRUE(bpe_instance->re->ok());  // compiled; if not, see re.error();

  std::string w;
  std::string text = "we'd annoyingly 顽皮";
  re2::StringPiece input(text);

  std::vector<std::string> v;
  while (RE2::FindAndConsume(&input, *bpe_instance->re, &w)) {
    v.push_back(w);
  }
  EXPECT_EQ(v, std::vector<std::string>({"we", "\'d", " annoyingly", " 顽皮"}));
}

TEST_F(TokenizerBPE, BytesToUnicodeConversion) {
  std::unordered_map<uint8_t, wchar_t> b2u;
  std::unordered_map<wchar_t, uint8_t> u2b;
  bpe_instance->bytes_to_unicode(&b2u, &u2b);

  // Validate the size of the maps
  EXPECT_EQ(b2u.size(), 256);
  EXPECT_EQ(u2b.size(), 256);

  // Check specific mappings to ensure they are correct
  EXPECT_EQ(b2u[0], 0x100);  // Assuming the mapping starts at 0x100 for 0
  EXPECT_EQ(u2b[0x100], 0);  // Reverse mapping check
}

TEST_F(TokenizerBPE, ByteEncodeToken) {
  std::unordered_map<uint8_t, wchar_t> b2u;
  bpe_instance->bytes_to_unicode(&b2u, NULL);
  std::wstring result;
  bpe_instance->byte_encode_token(" very", b2u, &result);

  EXPECT_EQ(bpe_instance->wstring_to_utf8(result), "Ġvery");
}

TEST_F(TokenizerBPE, LoadVocab) {
  auto &t2i = bpe_instance->t2i;
  auto &i2t = bpe_instance->i2t;

  EXPECT_GT(t2i.size(), 0);
  EXPECT_GT(i2t.size(), 0);

  // Check specific mappings to ensure they are correct
  // This assumes specific entries exist; adjust according to actual vocab content
  int expected_id_for_token = 42;
  std::string expected_token = "K";
  EXPECT_EQ(t2i[expected_token], expected_id_for_token);
  EXPECT_EQ(i2t[expected_id_for_token], expected_token);

  // Check consistency between maps
  for (const auto &pair : t2i) {
    auto token = pair.first;
    auto id = pair.second;
    EXPECT_EQ(i2t[id], token);
  }
}

TEST_F(TokenizerBPE, LoadMergeRules) {
  EXPECT_EQ(bpe_instance->bpe_ranks.size(), 50000);

  auto
      iter = bpe_instance->bpe_ranks.find({bpe_instance->utf8_to_wstring("Ġg"), bpe_instance->utf8_to_wstring("azed")});
  EXPECT_NE(iter, bpe_instance->bpe_ranks.end());
  EXPECT_EQ(iter->second, 49999);
}

TEST_F(TokenizerBPE, GetPairs) {
  std::vector<std::pair<std::wstring, std::wstring>> pairs;
  bpe_instance->get_pairs(bpe_instance->utf8_to_wstring("very"), &pairs);

  EXPECT_EQ(pairs.size(), 3);
  EXPECT_EQ(bpe_instance->wstring_to_utf8(pairs[1].first), "e");
  EXPECT_EQ(bpe_instance->wstring_to_utf8(pairs[1].second), "r");
}

TEST_F(TokenizerBPE, BPEAlgorithm) {
  EXPECT_EQ(bpe_instance->bpe_ranks.size(), 50000);

  std::vector<std::wstring> result;
  bpe_instance->bpe(bpe_instance->utf8_to_wstring("annoyingly"), &result);
  EXPECT_EQ(result, std::vector<std::wstring>({L"ann", L"oy", L"ingly"}));
}

TEST_F(TokenizerBPE, Tokenize) {
  EXPECT_TRUE(bpe_instance->re->ok());  // compiled; if not, see re.error();

  std::unordered_map<uint8_t, wchar_t> b2u;
  bpe_instance->bytes_to_unicode(&b2u, nullptr);

  std::vector<std::string> candidates = {
      "this is <|endoftext|> else<|endoftext|>",
      "<|endoftext|> else<|endoftext|>", "this is <|endoftext|> else",
      "this is <|endoftext|>else", "this is else"};
  auto _print_string_vec = [](std::vector<std::string> &v) {
    // To be compatible with Python's print(*lst, sep=', ')
    for (int i = 0; i < v.size(); ++i) {
      std::cout << v[i];
      if (i < v.size() - 1) std::cout << ", ";
    }
    std::cout << std::endl;
  };

  for (const auto &s : candidates) {
    std::vector<std::string> result;
    bpe_instance->tokenize(s, b2u, &result);
    _print_string_vec(result);
  }
}

TEST_F(TokenizerBPE, EncodeDecode) {
  std::unordered_map<uint8_t, wchar_t> b2u;
  std::unordered_map<wchar_t, uint8_t> u2b;
  bpe_instance->bytes_to_unicode(&b2u, &u2b);

  std::vector<std::string> candidates = {
      "this is <|endoftext|> else<|endoftext|>",
      "<|endoftext|> else<|endoftext|>", "this is <|endoftext|> else",
      "this is <|endoftext|>else", "this is else"};
  for (auto s : candidates) {
    std::vector<int> ids;
    bpe_instance->encode(s, b2u, &ids);
    EXPECT_GT(ids.size(), 0);
    EXPECT_EQ(bpe_instance->decode(ids, u2b), s);
  }
}
}
