#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <cassert>

namespace tokenizer {
class CharLevel final {
public:
    CharLevel(const std::string& text = "");

    [[nodiscard]] std::vector<int> encode(const std::string& text) const;

    [[nodiscard]] std::string decode(const std::vector<int>& ids) const;

private:
    std::unordered_map<char, int> stoi_;
    std::unordered_map<int, char> itos_;
};
}  // namespace tokenizer
