#include <memory>
#include <stdexcept>
#include <torch/torch.h>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <cstdlib>
#include <sstream>

#include "model.h"
#include "tokenizer/char.h"
#include "tokenizer/bpe.h"
#include "train.h"

int handle_train_command(const std::filesystem::path &config_path);
int handle_sample_command(const std::filesystem::path &checkpoint_dir);

int main(int argc, char **argv) {
  const std::string usage = R"(Usage: strgpt <command> <path>

Commands:
  train   /path/to/config.json       Train the model with the specified config file
  sample  /path/to/checkpoint/dir    Sample from the model using the specified checkpoint directory

Examples:
  strgpt train  ./config.json
  strgpt sample ./out/checkpoint_250_loss_2.686582
)";

  if (argc == 1) {
    std::cout << usage << std::endl;
    return 1;
  }
  if (argc != 3) {
    std::cerr << "Error: insufficient arguments provided\n\n" << usage << std::endl;
    return 1;
  }

  std::string command = argv[1];
  std::filesystem::path path = argv[2];

  if (command == "train") {
    return handle_train_command(path);
  } else if (command == "sample") {
    return handle_sample_command(path);
  } else {
    std::cerr << "Error: unknown command\n\n" << usage << std::endl;
    return 1;
  }
}

int handle_train_command(const std::filesystem::path &config_path) {
  // Build config
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

  return 0;
}

std::string get_editor_from_env() {
  const char *editor = std::getenv("EDITOR");
  return editor ? editor : "vi";
}

std::string get_prompt_from_editor() {
  std::string editor = get_editor_from_env();
  std::filesystem::path temp_file = std::filesystem::temp_directory_path() / "temp_prompt.txt";

  std::ofstream(temp_file).close(); // create the temporary file

  // Build the command to open the editor
  std::string command = editor + " " + temp_file.string();
  int ret = std::system(command.c_str());

  if (ret != 0) {
    std::cerr << "Error: Editor returned a non-zero exit code" << std::endl;
    return "";
  }

  // Read the content of the temporary file
  std::ifstream temp_file_stream(temp_file);
  std::stringstream buffer;
  buffer << temp_file_stream.rdbuf();

  // Remove the temporary file
  std::filesystem::remove(temp_file);

  return buffer.str();
}

int handle_sample_command(const std::filesystem::path &checkpoint_dir) {
  torch::Device device = torch::kMPS;
  train::Checkpoint ckpt{checkpoint_dir, device, false};
  auto &model = ckpt.model;

  // Initialize the environment based on the provided configuration
  torch::manual_seed(1337);
  std::cout << "Configuration and environment setup complete." << std::endl;

  // Read prompt from the favorite text editor,
  // like Git does when you write a commit message
  std::string prompt = "Hello, world!"; //get_prompt_from_editor();
  if (prompt.empty()) {
    std::cerr << "Error: No prompt provided" << std::endl;
    return 1;
  }

  // Build the tokenizer
  std::unique_ptr<tokenizer::Tokenizer> tok;
  if (ckpt.config.data.dataset.find("char") != std::string::npos) {
    std::cout << "Using char-level tokenizer" << std::endl;
    tok = std::make_unique<tokenizer::CharLevel>(prompt);
  } else {
    const char *tokenizer_config = std::getenv("TOKENIZER_CONFIG");
    if (tokenizer_config == nullptr) {
      throw std::runtime_error("Error: environment variable TOKENIZER_CONFIG is not set");
    }

    std::stringstream config_path;
    config_path << tokenizer_config << "/tokenizer.json";
    std::ifstream config_file(config_path.str(), std::ios::in);

    tok = std::make_unique<tokenizer::BPE>(config_file);
  }

  // Tokenize the prompt
  std::vector<int> tokens_int = tok->encode(prompt);
  std::cout << "Prompt: " << prompt << std::endl
            << "Tokens: ";
  for (const int token : tokens_int) {
    std::cout << token << ' ';
  }
  std::cout << std::endl;

  // Convert tokens to tensor
  // Note: from_blob does not take ownership of the data, so the data must stay in scope
  auto X =
      torch::from_blob(tokens_int.data(), {static_cast<long long>(tokens_int.size())}, torch::kInt64)
        .clone()
        .to(torch::kCPU)
        .contiguous();

  std::vector<int64_t> x_list(X.data_ptr<int64_t>(), X.data_ptr<int64_t>() + X.numel());
  std::cout << x_list << std::endl;

  // Run inference on the model
  constexpr int num_samples = 10; // num of samples to draw
  constexpr int max_new_tokens = 20; // number of tokens generated in each sample
  constexpr double temperature = 0.8; // 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
  constexpr int top_k = 200; // retain only the top_k most likely tokens, clamp others to have 0 probability
  model->eval();
  for (size_t k = 0; k < num_samples; k++) {
    auto Y = model->generate(X, max_new_tokens, temperature, top_k);
//    std::vector<int64_t> y_list(Y[0].data_ptr<double>(), Y[0].data_ptr<int64_t>() + Y[0].numel());
    Y = Y.contiguous();
    auto p = Y.options();
    std::cout << p << std::endl;
    auto tensor_data_ptr = Y.data_ptr<int64_t>();
    std::vector<int64_t> vec(tensor_data_ptr, tensor_data_ptr + Y.numel());
    std::cout << "Out " << k << ": " << vec << std::endl;
  }

  return 0;
}

