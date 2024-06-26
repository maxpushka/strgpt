#pragma once

#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "tokenizer/tokenizer.h"

namespace tokenizer {
class CharLevel final : public Tokenizer {
 public:
  CharLevel(const std::string& text = "");

  [[nodiscard]] std::vector<int> encode(const std::string& text) const;

  [[nodiscard]] std::string decode(const std::vector<int>& ids) const;

 private:
  std::unordered_map<char, int> stoi_;
  std::unordered_map<int, char> itos_;
};
}  // namespace tokenizer
