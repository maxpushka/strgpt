#include "bpe.h"
#include <memory>
#include <sstream>
#include <torch/torch.h>
#include <iostream>
#include <fstream>

int main() {
  const char* assets_root= std::getenv("ASSETS_ROOT");
  if (assets_root == nullptr) {
    std::cerr << "Error: ASSETS_ROOT env variable is not set" << std::endl;
    return 1;
  }

  std::stringstream vocab_path;
  vocab_path << assets_root << "/vocab.txt";
  std::fstream vocab_file(vocab_path.str(), std::ios::in);

  std::stringstream merges_path;
  merges_path << assets_root << "/merges.txt";
  std::fstream merges_file(merges_path.str(), std::ios::in);
  auto re = std::make_unique<RE2>(
    "('s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| "
    "?[^\\s\\p{L}\\p{N}]+|\\s+\\(?!\\S\\)|\\s+)");

  bpe::BPE tokenizer{vocab_file, merges_file, std::move(re)};

  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;
}

