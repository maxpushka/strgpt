#include "tokenizer/bpe.h"

#include <fstream>
#include <memory>
#include <string>

#include "gtest/gtest.h"

namespace tokenizer {
class TokenizerBPE : public testing::Test {
 protected:
  std::unique_ptr<tokenizer::BPE> bpe_tokenizer;

  void SetUp() override {
    const char *tokenizer_config = std::getenv("TOKENIZER_CONFIG");
    ASSERT_TRUE(tokenizer_config != nullptr);

    std::stringstream config_path;
    config_path << tokenizer_config << "/tokenizer.json";
    std::ifstream config_file(config_path.str(), std::ios::in);

    this->bpe_tokenizer = std::make_unique<tokenizer::BPE>(config_file);
  }

  void TearDown() override {}
};

TEST_F(TokenizerBPE, RegexCompilation) {
  std::string text = "we'd annoyingly 顽皮";
  std::smatch match;
  std::string input = text;

  std::vector<std::string> v;
  while (std::regex_search(input, match, bpe_tokenizer->re_)) {
    v.push_back(match.str());
    input = match.suffix().str();
  }
  EXPECT_EQ(v, std::vector<std::string>({"we", "'d", " annoyingly", " 顽皮"}));
}

TEST_F(TokenizerBPE, BytesToUnicodeConversion) {
  // Validate the size of the maps
  EXPECT_EQ(bpe_tokenizer->b2u_.size(), 256);
  EXPECT_EQ(bpe_tokenizer->u2b_.size(), 256);

  // Check specific mappings to ensure they are correct
  EXPECT_EQ(bpe_tokenizer->b2u_[0],
            0x100);  // Assuming the mapping starts at 0x100 for 0
  EXPECT_EQ(bpe_tokenizer->u2b_[0x100], 0);  // Reverse mapping check
}

TEST_F(TokenizerBPE, ByteEncodeToken) {
  std::wstring result = bpe_tokenizer->byte_encode_token(" very");
  EXPECT_EQ(bpe_tokenizer->wstring_to_utf8(result), "Ġvery");
}

TEST_F(TokenizerBPE, LoadVocab) {
  auto &t2i = bpe_tokenizer->t2i_;
  auto &i2t = bpe_tokenizer->i2t_;

  EXPECT_GT(t2i.size(), 0);
  EXPECT_GT(i2t.size(), 0);

  // Check specific mappings to ensure they are correct
  // This assumes specific entries exist.
  // Adjust according to actual vocab content.
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
  EXPECT_EQ(bpe_tokenizer->bpe_ranks_.size(), 50000);

  auto iter =
      bpe_tokenizer->bpe_ranks_.find({bpe_tokenizer->utf8_to_wstring("Ġg"),
                                      bpe_tokenizer->utf8_to_wstring("azed")});
  EXPECT_NE(iter, bpe_tokenizer->bpe_ranks_.end());
  EXPECT_EQ(iter->second, 49998);
}

TEST_F(TokenizerBPE, GetPairs) {
  auto pairs = bpe_tokenizer->get_pairs(bpe_tokenizer->utf8_to_wstring("very"));

  EXPECT_EQ(pairs.size(), 3);
  EXPECT_EQ(bpe_tokenizer->wstring_to_utf8(pairs[1].first), "e");
  EXPECT_EQ(bpe_tokenizer->wstring_to_utf8(pairs[1].second), "r");
}

TEST_F(TokenizerBPE, BPEAlgorithm) {
  EXPECT_EQ(bpe_tokenizer->bpe_ranks_.size(), 50000);

  std::vector<std::wstring> result =
      bpe_tokenizer->bpe(bpe_tokenizer->utf8_to_wstring("annoyingly"));
  EXPECT_EQ(result, std::vector<std::wstring>({L"ann", L"oy", L"ingly"}));
}

TEST_F(TokenizerBPE, Tokenize) {
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
    std::vector<std::string> result = bpe_tokenizer->tokenize(s);
    _print_string_vec(result);
  }
}

TEST_F(TokenizerBPE, EncodeDecode) {
  std::vector<std::string> candidates = {
      "this is <|endoftext|> else<|endoftext|>",
      "<|endoftext|> else<|endoftext|>", "this is <|endoftext|> else",
      "this is <|endoftext|>else", "this is else"};
  for (auto s : candidates) {
    std::vector<int> ids = bpe_tokenizer->encode(s);
    EXPECT_GT(ids.size(), 0);
    EXPECT_EQ(bpe_tokenizer->decode(ids), s);
  }
}

TEST_F(TokenizerBPE, HelloWorld) {
  std::string text = "Hello, world!";
  std::vector<int> encoded = bpe_tokenizer->encode(text);
  std::vector<int> expected = {15496, 11, 995, 0};

  ASSERT_EQ(encoded.size(), 4);
  EXPECT_EQ(encoded, expected);
}
}  // namespace tokenizer
