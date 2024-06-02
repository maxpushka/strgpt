#pragma once

#include <vector>
#include <string>
#include <filesystem>

namespace command {
void do_train(const std::filesystem::path &config_path);
void do_sample(const std::filesystem::path &checkpoint_dir);
}  // namespace command
