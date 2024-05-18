#include <filesystem>
#include <limits>
#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <string>
#include <random>
#include <regex>

#include "gradscaler.h"
#include "train.h"
#include "mapped_file.h"
#include "model.h"

namespace train {
// struct Checkpoint {
//   model::GPT model;
//   std::shared_ptr<torch::optim::Adam> optimizer;
//   Config config;
//   size_t iter_num;
//   double best_val_loss;

//   // Serialization function for torch::save
//   template <typename Archive>
//   void serialize(Archive& archive) {
//       archive(model, optimizer, config, iter_num, best_val_loss);
//   }
// };

// Save a checkpoint of the model
void save_checkpoint(const std::string &path,
                     std::shared_ptr<model::GPT> model,
                     std::shared_ptr<torch::optim::Optimizer> optimizer,
                     int iteration,
                     double loss) {
  std::string suffix = "_checkpoint_" + std::to_string(iteration) + "_loss_" + std::to_string(loss) + ".pt";
  torch::save(model, path + "/model" + suffix);
  torch::save(*optimizer, path + "/optimizer" + suffix);
}

// Function to find the latest checkpoint files in the given directory with matching versions
std::tuple<std::string, std::string, int> find_latest_matching_checkpoints(const std::string &directory) {
  std::regex model_pattern(R"(model_checkpoint_(\d+)_loss_\d+\.\d+\.pt)");
  std::regex optimizer_pattern(R"(optimizer_checkpoint_(\d+)_loss_\d+\.\d+\.pt)");
  std::map<int, std::string> model_files;
  std::map<int, std::string> optimizer_files;

  int max_version = -1;
  std::string latest_model_path;
  std::string latest_optimizer_path;

  namespace fs = std::filesystem;
  for (const auto &entry : fs::directory_iterator(directory)) {
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
  for (const auto &[version, model_path] : model_files) {
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
size_t load_checkpoint(const std::string &path,
                       std::shared_ptr<model::GPT> model,
                       std::shared_ptr<torch::optim::Optimizer> optimizer) {
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

struct Dataset {
  MappedFile train;
  MappedFile eval;

  int batch_size;
  int64_t block_size;
  torch::Device device;
};

std::pair<torch::Tensor, torch::Tensor> get_batch(const Dataset &dataset, const bool eval_mode) {
  const MappedFile &file = (eval_mode) ? dataset.eval : dataset.train;

  // Ensure the file has enough elements
  size_t num_elements = file.size() / sizeof(uint16_t);
  if (num_elements < static_cast<size_t>(dataset.block_size + 1)) {
    throw std::runtime_error("File size is too small for the specified block size.");
  }

  // Generate random indices for the batch
  std::vector<int> indices(dataset.batch_size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(0, num_elements - dataset.block_size - 1);

  // Prepare containers for input and target tensors
  std::vector<torch::Tensor> inputs, targets;
  inputs.reserve(dataset.batch_size);
  targets.reserve(dataset.batch_size);

  // Access the memory-mapped data
  const uint16_t *data = reinterpret_cast<const uint16_t *>(file.data());

  // Create tensors for each sample in the batch
  for (int i = 0; i < dataset.batch_size; ++i) {
    int idx = distrib(gen);
    const uint16_t *start = data + idx;
    const uint16_t *end = start + dataset.block_size + 1;

    // Create input and target tensors from slices of the mapped data
    std::vector<int64_t> input_data(start, end - 1);
    std::vector<int64_t> target_data(start + 1, end);
    inputs.push_back(torch::from_blob(input_data.data(), {dataset.block_size}, torch::kInt64).clone());
    targets.push_back(torch::from_blob(target_data.data(), {dataset.block_size}, torch::kInt64).clone());
  }

  // Stack all samples into a single tensor and transfer to the specified device
  auto X = torch::stack(inputs);
  auto Y = torch::stack(targets);
  if (dataset.device == torch::kCUDA) {
    // When memory is pinned, the operating system guarantees
    // that it will not be swapped out to disk.
    // This allows for more efficient and faster direct memory access (DMA)
    // transfers between the host and the GPU.
    X = X.pin_memory().to(dataset.device, /*non_blocking=*/true);
    Y = Y.pin_memory().to(dataset.device, /*non_blocking=*/true);
  } else {
    X = X.to(dataset.device);
    Y = Y.to(dataset.device);
  }

  return {X, Y};
}

std::map<std::string, double> estimate_loss(const int eval_iters, std::shared_ptr<model::GPT> model, const Dataset &dataset) {
  std::map<std::string, double> out;
  model->eval(); // Set model to evaluation mode

  for (const auto &split : {"train", "val"}) {
    std::vector<double> losses(eval_iters);

    for (int k = 0; k < eval_iters; ++k) {
      auto [X, Y] = get_batch(dataset, std::strcmp(split, "val") == 0);
      auto [logits, loss] = model->forward(X, Y);
      losses[k] = loss.item<double>();
    }

    double sum = std::accumulate(losses.begin(), losses.end(), 0.0);
    out[split] = sum / eval_iters;
  }

  model->train(); // Set model back to training mode
  return out;
}

double get_lr(const Config &c, int iter) {
  if (c.decay_lr) return c.learning_rate;

  // 1) linear warmup for warmup_iters steps
  if (iter < c.warmup_iters) return c.learning_rate * iter / c.warmup_iters;

  // 2) if it > lr_decay_iters, return min learning rate
  if (iter > c.lr_decay_iters) return c.min_lr;

  // 3) in between, use cosine decay down to min learning rate
  double decay_ratio = static_cast<double>(iter - c.warmup_iters) / (c.lr_decay_iters - c.warmup_iters);
  assert(0 <= decay_ratio && decay_ratio <= 1);
  double coeff = 0.5 * (1.0 + std::cos(M_PI * decay_ratio)); // coeff ranges 0..1
  return c.min_lr + coeff * (c.learning_rate - c.min_lr);
}

// The modified training function with checkpointing and learning rate adjustment
void train_model(std::shared_ptr<model::GPT> model,
                 std::shared_ptr<torch::optim::Optimizer> optimizer,
                 const Config &cfg,
                 size_t prev_iters_count,
                 torch::Device device) {
  // Read the dataset
  const Dataset dataset{
      .train = std::move(MappedFile{cfg.data_dir + "/" + "train" + ".bin"}),
      .eval = std::move(MappedFile{cfg.data_dir + "/" + "val" + ".bin"}),
      .batch_size = cfg.batch_size,
      .block_size = cfg.model.block_size,
      .device = device,
  };

  using clock = std::chrono::high_resolution_clock;
  auto t0 = std::chrono::high_resolution_clock::now();
  auto best_val_loss = std::numeric_limits<float>::max();
  double running_mfu = -1;
  size_t local_iter_num = 0;
  torch::amp::GradScaler scaler{torch::amp::GradScalerOptions{}.enabled(device == torch::kCUDA)};

  for (size_t iter_num = 0; iter_num < cfg.max_iters; ++iter_num) {
    // Determine and set the learning rate for this iteration
    double lr = get_lr(cfg, iter_num);
    for (auto &param_group : optimizer->param_groups()) {
      static_cast<torch::optim::AdamOptions &>(param_group.options()).lr(lr);
    }

    // Evaluate the loss on train/val sets and write checkpoints
    if (iter_num % cfg.eval_interval == 0) {
      auto losses = estimate_loss(cfg.eval_iters, model, dataset);
      std::cout << "step " << iter_num << ": train loss " << std::fixed << std::setprecision(4)
                << losses["train"] << ", val loss " << losses["val"] << std::endl;
      if (losses["val"] < best_val_loss || cfg.always_save_checkpoint) {
        best_val_loss = losses["val"];
        if (iter_num > 0) {
          // Save checkpoint
          std::cout << "saving checkpoint to " << cfg.out_dir << std::endl;
          // auto checkpoint = Checkpoint{
          //   .model = *model,
          //   .optimizer = std::dynamic_pointer_cast<torch::optim::Adam>(optimizer),
          //   .config = cfg,
          //   .iter_num = iter_num,
          //   .best_val_loss = best_val_loss,
          // };
          // torch::save(checkpoint, cfg.out_dir + "/ckpt.pt");
        }
      }
    }
    if (iter_num == 0 && cfg.eval_only) break;

    // Forward backward update, with optional gradient accumulation
    torch::Tensor step_loss;
    for (int micro_step = 0; micro_step < cfg.gradient_accumulation_steps; ++micro_step) {
      auto [X, Y] = get_batch(dataset, false);
      auto [logits, loss] = model->forward(X, Y);
      loss = loss / cfg.gradient_accumulation_steps;
      // Backward pass, with gradient scaling if training in fp16
      scaler.scale(loss).backward();
      step_loss = std::move(loss);
    }

    // Clip the gradient
    if (cfg.grad_clip != 0.0) {
      scaler.unscale_(optimizer);
      torch::nn::utils::clip_grad_norm_(model->parameters(), cfg.grad_clip);
    }

    // Step the optimizer and scaler if training in fp16
    scaler.step(optimizer);
    scaler.update();
    // Flush the gradients
    optimizer->zero_grad(true);

    // Timing and logging
    auto t1 = std::chrono::high_resolution_clock::now();
    float dt = std::chrono::duration<float>(t1 - t0).count();
    t0 = t1;
    if (iter_num % cfg.log_interval == 0) {
      float lossf = step_loss.item<float>() * cfg.gradient_accumulation_steps;
      if (local_iter_num >= 5) {
        double mfu = model->estimate_mfu(cfg.batch_size * cfg.gradient_accumulation_steps, dt);
        running_mfu = running_mfu == -1.0 ? mfu : 0.9 * running_mfu + 0.1 * mfu;
      }
      std::cout << "iter " << iter_num << ": loss " << std::fixed << std::setprecision(4) << lossf
                << ", time " << dt * 1000 << "ms, mfu " << running_mfu * 100 << "%" << std::endl;
    }
    iter_num++;
    local_iter_num++;
  }
}
}
