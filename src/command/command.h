#pragma once

#include <string>

#include "torch/torch.h"

namespace command {
void train_model(const std::string &config_path);
void sample_model(const std::string &checkpoint_dir, torch::Device device);
}  // namespace command
