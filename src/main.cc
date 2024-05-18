#include <torch/torch.h>
#include <iostream>
#include <filesystem>
#include <fstream>

#include "model.h"
#include "train.h"

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Error: config path is not provided\n\n"
              << "Usage: strgpt /path/to/config/file" << std::endl;
    return 1;
  }

  // Build config
  std::filesystem::path config_path{argv[1]};
  if (!std::filesystem::exists(config_path)) {
    std::cerr << "Error: config file does not exist at a given path: " << config_path << std::endl;
    return 1;
  }
  std::ifstream config_file{config_path};
  nlohmann::json config_json = nlohmann::json::parse(config_file,
    /* callback */ nullptr,
    /* allow_exceptions */ true,
    /* ignore_comments */ true);
  train::Config config = config_json.get<train::Config>();

  // Initialize the environment based on the provided configuration
  std::filesystem::create_directories(config.out_dir);
  torch::manual_seed(1337);
  torch::Device device{config.device};
  std::cout << "Configuration and environment setup complete." << std::endl;

  // Initialize the model
  auto model = std::make_shared<model::GPT>(config.model);
  model->to(device);
  std::cout << "Model initialized." << std::endl;

  // Initialize the optimizer
  auto options = torch::optim::AdamOptions(config.learning_rate)
      .betas({config.beta1, config.beta2})
      .weight_decay(config.weight_decay);
  auto optimizer = std::make_shared<torch::optim::Adam>(model->parameters(), options);
  std::cout << "Optimizer initialized." << std::endl;

  // Load weights from a checkpoint
  size_t previous_iterations_count = 0;
  // if (config.init_from == "resume") {
  //   previous_iterations_count = train::load_checkpoint(config.out_dir, model, optimizer);
  // }

  // Run training loop
  std::cout << "Starting training loop." << std::endl;
  auto start = std::chrono::high_resolution_clock::now();
  train::train_model(model, optimizer, config, previous_iterations_count, device);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = duration_cast<std::chrono::seconds>(end - start).count();
  std::cout << "Training completed in " << duration << "s" << std::endl;

  return 0;
}
