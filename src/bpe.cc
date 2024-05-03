#include "bpe.h"
#include <codecvt>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <memory>
#include <locale>
#include <iostream>

namespace bpe {
BPE::BPE(std::istream &config_file, std::unique_ptr<RE2> re) : re(std::move(re)) {
  using json = nlohmann::json;

  json vocab, merges;
  try {
    json config = json::parse(config_file);
    vocab = config.at("model").at("vocab");
    merges = config.at("model").at("merges");
  } catch (json::parse_error &e) {
    std::cerr << "Failed to parse tokenizer config: " << e.what() << std::endl;
  } catch (json::out_of_range &e) {
    std::cerr << "JSON key error: " << e.what() << std::endl;
  } catch (json::exception &e) {
    std::cerr << "General JSON error: " << e.what() << std::endl;
  }

  load_vocab(vocab);
  load_merge_rules(merges);
  load_bytes_to_unicode();
}

std::vector<int> BPE::encode(const std::string &text) const {
  std::vector<std::string> result = tokenize(text);

  std::vector<int> ids;
  ids.reserve(result.size());
  for (const auto &s : result) {
    ids.push_back(t2i.at(s));
  }

  return ids;
}

std::string BPE::decode(const std::vector<int> &ids) const {
  std::string concat;
  for (const int &id : ids) {
    concat += i2t.at(id);
  }

  std::wstring w = utf8_to_wstring(concat);
  std::string r;
  for (const wchar_t &c : w) {
    r.push_back(char(u2b.at(c)));
  }
  return r;
}

void BPE::load_vocab(const nlohmann::json &ins) {
  for (const auto &[key, value] : ins.items()) {
    t2i.insert({key, value});
    i2t.insert({value, key});
  }
}

void BPE::load_merge_rules(const nlohmann::json &ins) {
  for (const auto &[key, value] : ins.items()) {
    std::string line = value.get<std::string>();
    int d = line.find(" ");  // merges.txt file use ASCII space
    bpe_ranks.try_emplace({utf8_to_wstring(line.substr(0, d)),
                           utf8_to_wstring(line.substr(d + 1))},
                          std::stoi(key) - 1);
  }
}

void BPE::load_bytes_to_unicode() {
  auto _insert_range = [=, this](int start, int end) {
    for (int c = start; c <= end; c++) {
      b2u.insert({uint8_t(c), wchar_t(c)});
    }
  };

  b2u.clear();
  _insert_range(L'!', L'~');
  _insert_range(L'¡', L'¬');
  _insert_range(L'®', L'ÿ');

  int n = 0;
  for (int b = 0; b < 256; b++) {
    if (b2u.find(uint8_t(b)) == b2u.end()) {
      b2u.insert({uint8_t(b), wchar_t(256 + n)});
      n++;
    }
  }

  u2b.clear();
  for (auto e : b2u) {
    u2b.insert({e.second, e.first});
  }
}

std::wstring BPE::utf8_to_wstring(const std::string &str) {
  std::wstring_convert<std::codecvt_utf8<wchar_t>> convert;
  return convert.from_bytes(str);
}

std::string BPE::wstring_to_utf8(const std::wstring &str) {
  std::wstring_convert<std::codecvt_utf8<wchar_t>> convert;
  return convert.to_bytes(str);
}

std::string BPE::utf8(wchar_t c) {
  std::wstring w(1, c);
  return wstring_to_utf8(w);
}

std::vector<std::string> BPE::tokenize(const std::string &text) const {
  const std::string eot("<|endoftext|>");
  size_t s = 0;
  size_t i = text.find(eot);
  std::vector<std::string> result;
  while (i != std::string::npos) {
    _tokenize(text.substr(s, i - s), result);
    result.push_back(eot);
    s = i + eot.size();
    i = text.find(eot, s);
  }
  _tokenize(text.substr(s), result);

  return result;
}

void BPE::_tokenize(const std::string &text, std::vector<std::string> &result) const {
  re2::StringPiece input(text);
  std::string token;

  while (RE2::FindAndConsume(&input, *re, &token)) {
    std::wstring wtoken = byte_encode_token(token);
    std::vector<std::wstring> bpe_tokens = bpe(wtoken);
    for (const auto &ws : bpe_tokens) {
      result.push_back(wstring_to_utf8(ws));
    }
  }
}

// Given a token as a UTF8 string, encode each byte into an wchar_t
std::wstring BPE::byte_encode_token(const std::string &token) const {
  std::wstring result;
  result.reserve(token.size());

  for (char c : token) {
    wchar_t wc = b2u.at(uint8_t(c));
    result.push_back(wc);
  }

  return result;
};

std::vector<std::wstring> BPE::bpe(const std::wstring &token) const {
  std::set<int> merged;  // records indices in pairs that were merged.
  auto _left = [](int i, std::set<int> &merged) {
    for (int j = i - 1; j >= -1; j--) {
      if (merged.find(j) == merged.end()) return j;
    }
    return -1;
  };
  auto _right = [](int i, int cap, std::set<int> &merged) {
    for (int j = i + 1; j < cap; j++) {
      if (merged.find(j) == merged.end()) return j;
    }
    return cap;
  };

  auto pairs = get_pairs(token);

  while (true) {
    int min_score = INT_MAX;
    int to_merge = -1;  // indices into pairs.

    for (int i = 0; i < pairs.size(); ++i) {
      if (merged.find(i) == merged.end()) {  // pair i is not merged.
        auto iter = bpe_ranks.find(pairs[i]);
        int score = iter != bpe_ranks.end() ? iter->second : INT_MAX;
        if (score < min_score) {
          min_score = score;
          to_merge = i;
        }
      }
    }

    if (to_merge == -1) break;

    merged.insert(to_merge);
    std::wstring merge_into = pairs[to_merge].first + pairs[to_merge].second;

    int l = _left(to_merge, merged);
    if (l >= 0) pairs[l].second = merge_into;
    int r = _right(to_merge, pairs.size(), merged);
    if (r < pairs.size()) pairs[r].first = merge_into;
  }  // end while (true)

  std::vector<std::wstring> result;
  if (merged.size() == pairs.size()) {
    result.push_back(token);
  } else {
    for (int i = 0; i < pairs.size(); ++i) {
      if (merged.find(i) == merged.end()) {
        if (_left(i, merged) < 0) result.push_back(pairs[i].first);
        result.push_back(pairs[i].second);
      }
    }
  }

  return result;
}

std::vector<std::pair<std::wstring, std::wstring>> BPE::get_pairs(const std::wstring &word) {
  std::vector<std::pair<std::wstring, std::wstring>> pairs;
  if (word.size() < 2) return pairs;

  wchar_t previous = word[0];
  for (size_t i = 1; i < word.size(); i++) {
    pairs.emplace_back(std::wstring(1, previous), std::wstring(1, word[i]));
    previous = word[i];
  }

  return pairs;
}
}  // namespace bpe
