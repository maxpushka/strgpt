#pragma once

#include <string>
#include <vector>

namespace tokenizer {
class Tokenizer {
 public:
  virtual ~Tokenizer() = default;

  [[nodiscard]] virtual std::vector<int> encode(
      const std::string& text) const = 0;
  [[nodiscard]] virtual std::string decode(
      const std::vector<int>& ids) const = 0;
};
}  // namespace tokenizer
