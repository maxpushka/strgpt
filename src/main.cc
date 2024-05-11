#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <filesystem>
#include <memory>

#include "model.h"
#include "train.h"

int main() {
  Config config;

  // Initialize the environment based on the provided configuration
  std::filesystem::create_directories(config.out_dir);
  torch::manual_seed(1337);
  torch::Device device{config.device};
  std::cout << "Configuration and environment setup complete." << std::endl;

  // Initialize the model
  auto model = std::make_shared<GPT>(config.model);
  model->to(device);

  // Initialize the optimizer
  auto options = torch::optim::AdamOptions(config.learning_rate)
      .betas({config.beta1, config.beta2})
      .weight_decay(config.weight_decay);
  auto optimizer = std::make_shared<torch::optim::Adam>(model->parameters(), options);

  // Load weights from a checkpoint
  size_t previous_iterations_count = 0;
  if (config.init_from == "resume") {
    previous_iterations_count = load_checkpoint(config.out_dir, model, optimizer);
  }
  std::cout << "Model initialized." << std::endl;

  // Run training loop
  std::cout << "Starting training loop." << std::endl;
  auto start = std::chrono::high_resolution_clock::now();
  train_model_with_scheduler_and_checkpointing(model, optimizer, config, previous_iterations_count, device);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;
  std::cout << "Training completed in " << diff.count() << " s" << std::endl;

  return 0;
}
