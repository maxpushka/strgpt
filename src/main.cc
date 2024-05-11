#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <filesystem>
#include <random>

#include "model.h"
#include "mapped_file.h"

// Configuration struct to hold all training parameters
struct Config {
  std::string data_dir = "data";
  std::string out_dir = "out";

  int eval_interval = 2000;
  int log_interval = 1;
  int eval_iters = 200;
  bool eval_only = false;
  bool always_save_checkpoint = true;
  std::string init_from = "scratch";
  std::string dataset = "openwebtext";
  int gradient_accumulation_steps = 40; // 5 * 8
  int batch_size = 64;
  GPTConfig model{
      .vocab_size = 50257,
      .block_size = 256, // context of up to 256 previous characters
      .n_layer = 6,
      .n_head = 6,
      .n_embd = 384,
      .dropout = 0.2,
      .bias = false,
  };
  float learning_rate = 6e-4;
  int max_iters = 600000;
  float weight_decay = 0.1;
  float beta1 = 0.9;
  float beta2 = 0.95;
  float grad_clip = 1.0;
  bool decay_lr = true;
  int warmup_iters = 2000;
  int lr_decay_iters = 600000;
  float min_lr = 6e-5;
  std::string backend = "nccl";
  std::string device = "cuda";
  std::string dtype = "float16"; // Assume float16 by default
  bool compile = true;
};

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

std::shared_ptr<torch::optim::Optimizer> configure_optimizer(std::shared_ptr<GPT> model, const Config &cfg) {
  // Create and return the AdamW optimizer
  auto options = torch::optim::AdamOptions(cfg.learning_rate)
      .betas({cfg.beta1, cfg.beta2})
      .weight_decay(cfg.weight_decay);
  return std::make_shared<torch::optim::Adam>(model->parameters(), options);
}

// Save a checkpoint of the model
void save_checkpoint(const std::string &path,
                     std::shared_ptr<GPT> model,
                     std::shared_ptr<torch::optim::Optimizer> optimizer,
                     int iteration) {
  torch::save(model, path + "/model_checkpoint_" + std::to_string(iteration) + ".pt");
  torch::save(*optimizer, path + "/optimizer_checkpoint_" + std::to_string(iteration) + ".pt");
}

// Load a checkpoint of the model
void load_checkpoint(const std::string &path, std::shared_ptr<GPT> model, torch::optim::Optimizer *optimizer) {
  torch::load(model, path + "/model_checkpoint.pt");
  torch::load(*optimizer, path + "/optimizer_checkpoint.pt");
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
void train_model_with_scheduler_and_checkpointing(std::shared_ptr<GPT> model,
                                                  std::shared_ptr<torch::optim::Optimizer> optimizer,
                                                  const Config &cfg,
                                                  torch::Device device) {
  for (int iter = 0; iter < cfg.max_iters; ++iter) {
    auto [X, Y] = get_batch(cfg.data_dir, "train", cfg.batch_size, cfg.model.block_size, device);
    model->train();
    optimizer->zero_grad();
    auto [logits, loss] = model->forward(X, Y);
    loss.backward();
    optimizer->step();

    adjust_learning_rate(optimizer, iter, cfg);

    if (iter % cfg.log_interval == 0) {
      std::cout << "Iteration " << iter << ": loss=" << loss.item<float>() << ", lr="
                << optimizer->param_groups().front().options().get_lr() << std::endl;
    }

    if ((iter % cfg.eval_interval == 0) || (iter == cfg.max_iters - 1)) {
      model->eval();
      auto [eval_X, eval_Y] = get_batch(cfg.data_dir, "val", cfg.batch_size, cfg.model.block_size, device);
      auto [eval_logits, eval_loss] = model->forward(eval_X, eval_Y);
      std::cout << "Eval Iteration " << iter << ": eval loss=" << eval_loss.item<float>() << std::endl;
      save_checkpoint(cfg.out_dir, model, optimizer, iter);
    }
  }
}

int main() {
  Config config; // Create a configuration instance with default values

  // Initialize the environment based on the provided configuration
  std::filesystem::create_directories(config.out_dir);
  torch::manual_seed(1337);
  torch::Device device{torch::kCPU};
  std::cout << "Configuration and environment setup complete." << std::endl;

  // Initialize the model
  auto model = std::make_shared<GPT>(config.model);
  model->to(device);
  std::cout << "Model initialized." << std::endl;

  std::cout << "Starting training loop." << std::endl;
  auto optimizer = configure_optimizer(model, config);
  auto start = std::chrono::high_resolution_clock::now();
  train_model_with_scheduler_and_checkpointing(model, optimizer, config, device);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;
  std::cout << "Training completed in " << diff.count() << " s" << std::endl;

  return 0;
}
