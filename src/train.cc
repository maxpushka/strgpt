#include <filesystem>
#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <string>
#include <random>
#include <regex>

#include "train.h"
#include "mapped_file.h"
#include "model.h"

namespace train {
// Save a checkpoint of the model
void save_checkpoint(const std::string &path,
                     std::shared_ptr<model::GPT> model,
                     std::shared_ptr<torch::optim::Optimizer> optimizer,
                     int iteration) {
  torch::save(model, path + "/model_checkpoint_" + std::to_string(iteration) + ".pt");
  torch::save(*optimizer, path + "/optimizer_checkpoint_" + std::to_string(iteration) + ".pt");
}

// Function to find the latest checkpoint files in the given directory with matching versions
std::tuple<std::string, std::string, int> find_latest_matching_checkpoints(const std::string& directory) {
    std::regex model_pattern("model_checkpoint_(\\d+)\\.pt");
    std::regex optimizer_pattern("optimizer_checkpoint_(\\d+)\\.pt");
    std::map<int, std::string> model_files;
    std::map<int, std::string> optimizer_files;

    int max_version = -1;
    std::string latest_model_path;
    std::string latest_optimizer_path;

    namespace fs = std::filesystem;
    for (const auto& entry : fs::directory_iterator(directory)) {
        std::smatch matches;
        const std::string filename = entry.path().filename().string();

        if (std::regex_search(filename, matches, model_pattern)) {
            int version = std::stoi(matches[1].str());
            model_files[version] = entry.path().string();
        } else if (std::regex_search(filename, matches, optimizer_pattern)) {
            int version = std::stoi(matches[1].str());
            optimizer_files[version] = entry.path().string();
        }
    }

    // Check for the highest version that exists in both maps
    for (const auto& [version, model_path] : model_files) {
        auto opt_it = optimizer_files.find(version);
        if (opt_it != optimizer_files.end()) {
            if (version > max_version) {
                max_version = version;
                latest_model_path = model_path;
                latest_optimizer_path = opt_it->second;
            }
        }
    }

    if (max_version != -1) {
        return {latest_model_path, latest_optimizer_path, max_version};
    } else {
        return {"", "", -1};
    }
}

// Load the latest checkpoint of the model and optimizer
size_t load_checkpoint(const std::string& path, std::shared_ptr<model::GPT> model, std::shared_ptr<torch::optim::Optimizer> optimizer) {
  auto [latest_model_path, latest_optimizer_path, latest_version] = find_latest_matching_checkpoints(path);
  if (latest_model_path.empty() || latest_optimizer_path.empty()) {
    std::cerr << "No checkpoints found in the directory: " << path << std::endl;
    return 0;
  }

  torch::load(model, latest_model_path);
  torch::load(*optimizer, latest_optimizer_path);
  std::cout << "Loaded model from " << latest_model_path << std::endl;
  std::cout << "Loaded optimizer from " << latest_optimizer_path << std::endl;
  return latest_version;
}

std::pair<torch::Tensor, torch::Tensor> get_batch(const std::string &data_dir,
                                                  const std::string &split,
                                                  int batch_size,
                                                  int block_size,
                                                  const torch::Device &device) {
  // Construct file path and create a memory-mapped file
  std::string file_path = data_dir + "/" + split + ".bin";
  MappedFile mapped_file(file_path);

  // Ensure the file has enough elements
  size_t num_elements = mapped_file.size() / sizeof(uint16_t);
  if (num_elements < static_cast<size_t>(block_size + 1)) {
    throw std::runtime_error("File size is too small for the specified block size.");
  }

  // Generate random indices for the batch
  std::vector<int> indices(batch_size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(0, num_elements - block_size - 1);

  // Prepare containers for input and target tensors
  std::vector<torch::Tensor> inputs, targets;
  inputs.reserve(batch_size);
  targets.reserve(batch_size);

  // Access the memory-mapped data
  const uint16_t *data = reinterpret_cast<const uint16_t *>(mapped_file.data());

  // Create tensors for each sample in the batch
  for (int i = 0; i < batch_size; ++i) {
    int idx = distrib(gen);
    const uint16_t *start = data + idx;
    const uint16_t *end = start + block_size + 1;

    // Create input and target tensors from slices of the mapped data
    std::vector<int64_t> input_data(start, end - 1);
    std::vector<int64_t> target_data(start + 1, end);
    inputs.push_back(torch::from_blob(input_data.data(), {block_size}, torch::kInt64).clone());
    targets.push_back(torch::from_blob(target_data.data(), {block_size}, torch::kInt64).clone());
  }

  // Stack all samples into a single tensor and transfer to the specified device
  auto X = torch::stack(inputs).to(device, /*non_blocking=*/true);
  auto Y = torch::stack(targets).to(device, /*non_blocking=*/true);

  return {X, Y};
}

// A simple learning rate scheduler function
void adjust_learning_rate(std::shared_ptr<torch::optim::Optimizer> optimizer, int current_iter, const Config &cfg) {
  double lr;
  if (current_iter < cfg.warmup_iters) {
    lr = cfg.learning_rate * static_cast<float>(current_iter) / static_cast<float>(cfg.warmup_iters);
  } else {
    const double
        decay_rate = static_cast<double>(current_iter - cfg.warmup_iters)
        / static_cast<double>(cfg.lr_decay_iters - cfg.warmup_iters);
    lr = cfg.min_lr + (cfg.learning_rate - cfg.min_lr) * (1.0f - std::cos(decay_rate * 3.14159265359f)) * 0.5f;
  }

  for (auto &group : optimizer->param_groups()) {
    group.options().set_lr(lr);
  }
}

// The modified training function with checkpointing and learning rate adjustment
void train_model_with_scheduler_and_checkpointing(std::shared_ptr<model::GPT> model,
                                                  std::shared_ptr<torch::optim::Optimizer> optimizer,
                                                  const Config &cfg,
                                                  size_t prev_iters_count,
                                                  torch::Device device) {
  using clock = std::chrono::high_resolution_clock;
  for (size_t session_iter = 0; session_iter < cfg.max_iters; ++session_iter) {
    auto start = clock::now();

    auto [X, Y] = get_batch(cfg.data_dir, "train", cfg.batch_size, cfg.model.block_size, device);
    model->train();
    optimizer->zero_grad();
    auto [logits, loss] = model->forward(X, Y);
    loss.backward();
    optimizer->step();

    size_t iter = session_iter+prev_iters_count;
    adjust_learning_rate(optimizer, iter, cfg);

    if (iter % cfg.log_interval == 0) {
      auto duration = duration_cast<std::chrono::milliseconds>(clock::now() - start).count();
      std::cout << "Iteration " << iter << ": loss=" << loss.item<float>()
                << ", lr=" << optimizer->param_groups().front().options().get_lr()
                << ", time=" << duration << "ms" << std::endl;
    }

    if ((iter % cfg.eval_interval == 0) || (session_iter == cfg.max_iters - 1)) {
      auto eval_start = clock::now();
      model->eval();
      auto [eval_X, eval_Y] = get_batch(cfg.data_dir, "val", cfg.batch_size, cfg.model.block_size, device);
      auto [eval_logits, eval_loss] = model->forward(eval_X, eval_Y);
      auto eval_duration = duration_cast<std::chrono::milliseconds>(clock::now() - eval_start).count();

      std::cout << "Eval Iteration " << iter << ": loss=" << eval_loss.item<float>()
                << ", time=" << eval_duration << "ms" << std::endl;
      save_checkpoint(cfg.out_dir, model, optimizer, iter);
    }
  }
}
}