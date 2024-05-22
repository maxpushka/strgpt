#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>

#include "tokenizer/char.h"

class TokenizerCharLevel : public testing::Test {
protected:
    std::unique_ptr<tokenizer::CharLevel> char_tokenizer;

    void SetUp() override {
        const std::string data = R"( !$&'",-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz)";
        char_tokenizer = std::make_unique<tokenizer::CharLevel>(data);
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
    std::vector<int> expected = { 32, 53, 0, 40, 43 };
    EXPECT_EQ(encoded, expected);
}

TEST_F(TokenizerCharLevel, DecodeSingleID) {
    std::vector<int> ids = {32};
    std::string decoded = char_tokenizer->decode(ids);
    EXPECT_EQ(decoded, "T");
}

TEST_F(TokenizerCharLevel, DecodeIDs) {
    std::vector<int> ids = { 32, 53, 0, 40, 43 };
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
    EXPECT_THROW(char_tokenizer->encode(text), std::runtime_error);
}

TEST_F(TokenizerCharLevel, DecodeUnknownID) {
    std::vector<int> ids = {9999};  // Assuming 9999 is not a valid ID
    EXPECT_THROW(char_tokenizer->decode(ids), std::runtime_error);
}
