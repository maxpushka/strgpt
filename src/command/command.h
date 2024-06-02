#pragma once

#include <string>

namespace command {
void do_train(const std::string &config_path);
void do_sample(const std::string &checkpoint_dir);
}  // namespace command
