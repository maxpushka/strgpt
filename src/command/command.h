#pragma once

#include <string>

#include "torch/torch.h"

namespace command {
void do_train(const std::string &config_path);
void do_sample(const std::string &checkpoint_dir, torch::Device device);
}  // namespace command
