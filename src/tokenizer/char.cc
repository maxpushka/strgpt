#include <sstream>
#include <stdexcept>

#include "tokenizer/char.h"

namespace tokenizer {
CharLevel::CharLevel(const std::string& text) {
    const std::array<std::string, 2> corpus {
        R"( !$&'",-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz)",
        text,
    };

    // Initialize the character to integer and integer to character mappings
    for (const std::string& text : corpus) {
        for (char ch : text) {
            if (stoi_.find(ch) != stoi_.end()) continue;
            int id = stoi_.size();
            stoi_[ch] = id;
            itos_[id] = ch;
        }
    }
}

[[nodiscard]] std::vector<int> CharLevel::encode(const std::string& text) const {
    std::vector<int> encoded;
    encoded.reserve(text.size());
    for (const char& ch : text) {
        auto it = stoi_.find(ch);
        if (it == stoi_.end()) {
            std::stringstream ss;
            ss << "Character not found in tokenizer vocabulary (`" << ch << "`)";
            throw std::runtime_error(ss.str());
        }
        encoded.push_back(it->second);
    }
    return encoded;
}

[[nodiscard]] std::string CharLevel::decode(const std::vector<int>& ids) const {
    std::string decoded;
    decoded.reserve(ids.size());
    for (const int& id : ids) {
        auto it = itos_.find(id);
        if (it == itos_.end()) {
            std::stringstream ss;
            ss << "ID not found in tokenizer vocabulary (" << id << ")";
            throw std::runtime_error(ss.str());
        }
        decoded.push_back(it->second);
    }
    return decoded;
}
}  // namespace tokenizer
