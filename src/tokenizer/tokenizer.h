#pragma once

#include <string>
#include <vector>

#include "nlohmann/json.hpp"

namespace tokenizer {
class Tokenizer {
 public:
  virtual ~Tokenizer() = default;

  [[nodiscard]] virtual std::vector<int> encode(
      const std::string& text) const = 0;
  [[nodiscard]] virtual std::string decode(
      const std::vector<int>& ids) const = 0;
  [[nodiscard]] virtual nlohmann::json dump_state() const = 0;
};

enum class Type {
  Char,
  BPE,
  Invalid = -1,
};
NLOHMANN_JSON_SERIALIZE_ENUM(Type, {{Type::Invalid, nullptr},
                                    {Type::Char, "char"},
                                    {Type::BPE, "bpe"}});
}  // namespace tokenizer
