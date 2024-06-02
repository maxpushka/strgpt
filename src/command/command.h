#pragma once

#include <string>

#include "tokenizer/tokenizer.h"
#include "torch/torch.h"

namespace command {
void train_model(const std::string &config_path);
void sample_model(const std::string &checkpoint_dir, torch::Device device);
void prepare_data(const std::string &dataset_url, const std::string &out_path,
                  const tokenizer::Type tok_type);
}  // namespace command
