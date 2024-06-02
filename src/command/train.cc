#include <iostream>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <torch/torch.h>

#include "command/command.h"
#include "model/train.h"
#include "model/model.h"

namespace command {
void do_train(const std::filesystem::path &config_path) {
  // Build config
  if (!std::filesystem::exists(config_path)) {
    throw std::runtime_error("Error: config file does not exist at a given path: " + config_path.string());
  }

  std::ifstream config_file{config_path};
  nlohmann::json config_json = nlohmann::json::parse(config_file,
      /* callback */ nullptr,
      /* allow_exceptions */ true,
      /* ignore_comments */ true);
  train::Config config = config_json.get<train::Config>();

  // Initialize the environment based on the provided configuration
  std::filesystem::create_directories(config.train.out_dir);
  torch::manual_seed(1337);
  torch::Device device{config.train.device};
  std::cout << "Configuration and environment setup complete." << std::endl;

  // Initialize the model
  auto model = std::make_shared<model::GPT>(config.model);
  model->to(device);
  std::cout << "Model initialized." << std::endl;

  // Initialize the optimizer
  auto options = torch::optim::AdamOptions(config.train.learning_rate)
      .betas({config.train.beta1, config.train.beta2})
      .weight_decay(config.train.weight_decay);
  auto optimizer = std::make_shared<torch::optim::Adam>(model->parameters(), options);
  std::cout << "Optimizer initialized." << std::endl;

  // Run training loop
  std::cout << "Starting training loop." << std::endl;
  auto start = std::chrono::high_resolution_clock::now();
  train::train_model(model, optimizer, config, device);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = duration_cast<std::chrono::seconds>(end - start).count();
  std::cout << "Training completed in " << duration << "s" << std::endl;
}
}  // namespace command
