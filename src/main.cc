#include "bpe.h"
#include <memory>
#include <sstream>
#include <torch/torch.h>
#include <iostream>
#include <fstream>

int main() {
  const char *assets_root = std::getenv("ASSETS_ROOT");
  if (assets_root == nullptr) {
    std::cerr << "Error: ASSETS_ROOT env variable is not set" << std::endl;
    return 1;
  }

  auto re = std::make_unique<RE2>(
      "('s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| "
      "?[^\\s\\p{L}\\p{N}]+|\\s+\\(?!\\S\\)|\\s+)");

  std::stringstream path;
  path << assets_root << "/tokenizer.json";
  std::ifstream config{path.str(), std::ifstream::in};
  bpe::BPE tok{config, std::move(re)};

  std::string text{"Hello, world!"};
  std::vector<int> ids = tok.encode(text);
  for (const auto &id : ids) {
    std::cout << id << " ";
  }
  std::cout << std::endl;

  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;
}
