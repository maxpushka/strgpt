#include "tokenizer/char.h"

#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"

class TokenizerCharLevel : public testing::Test {
 protected:
  std::unique_ptr<tokenizer::CharLevel> char_tokenizer;

  void SetUp() override {
    char_tokenizer = std::make_unique<tokenizer::CharLevel>();
  }

  void TearDown() override {}
};

TEST_F(TokenizerCharLevel, EncodeSingleCharacter) {
  std::string text = "T";
  std::vector<int> encoded = char_tokenizer->encode(text);
  ASSERT_EQ(encoded.size(), 1);
  EXPECT_EQ(encoded[0], 32);
}

TEST_F(TokenizerCharLevel, EncodeString) {
  std::string text = "To be";
  std::vector<int> encoded = char_tokenizer->encode(text);
  std::vector<int> expected = {32, 53, 0, 40, 43};
  EXPECT_EQ(encoded, expected);
}

TEST_F(TokenizerCharLevel, DecodeSingleID) {
  std::vector<int> ids = {32};
  std::string decoded = char_tokenizer->decode(ids);
  EXPECT_EQ(decoded, "T");
}

TEST_F(TokenizerCharLevel, DecodeIDs) {
  std::vector<int> ids = {32, 53, 0, 40, 43};
  std::string decoded = char_tokenizer->decode(ids);
  EXPECT_EQ(decoded, "To be");
}

TEST_F(TokenizerCharLevel, EncodeDecodeConsistency) {
  std::string text = "To be, or not to be";
  std::vector<int> encoded = char_tokenizer->encode(text);
  std::string decoded = char_tokenizer->decode(encoded);
  EXPECT_EQ(decoded, text);
}

TEST_F(TokenizerCharLevel, EncodeUnknownCharacter) {
  std::string text = "@";  // Assuming '@' is not in the initial data set
  // Basically, casting to void tells the compiler
  // "Yes I know I'm discarding this, yes I'm sure of it."
  // Just a hacky way to intentionally discard a [[nodiscard]] value
  EXPECT_THROW(static_cast<void>(char_tokenizer->encode(text)), std::runtime_error);
}

TEST_F(TokenizerCharLevel, DecodeUnknownID) {
  std::vector<int> ids = {9999};  // Assuming 9999 is not a valid ID
  // Basically, casting to void tells the compiler
  // "Yes I know I'm discarding this, yes I'm sure of it."
  // Just a hacky way to intentionally discard a [[nodiscard]] value
  EXPECT_THROW(static_cast<void>(char_tokenizer->decode(ids)), std::runtime_error);
}
