#include <iostream>
#include <fstream>
#include <stdexcept>
#include <torch/torch.h>

#include "command/command.h"
#include "tokenizer/char.h"
#include "tokenizer/bpe.h"
#include "model/train.h"

namespace command {
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
    throw std::runtime_error("Error: editor returned a non-zero exit code");
  }

  // Read the content of the temporary file
  std::ifstream temp_file_stream(temp_file);
  std::stringstream buffer;
  buffer << temp_file_stream.rdbuf();

  // Remove the temporary file
  std::filesystem::remove(temp_file);

  return buffer.str();
}

void do_sample(const std::filesystem::path &checkpoint_dir) {
  torch::Device device = torch::kMPS; // TODO: move to command options
  train::Checkpoint ckpt{checkpoint_dir, device, false};
  auto &model = ckpt.model;

  // Initialize the environment based on the provided configuration
  torch::manual_seed(1337);
  std::cout << "Configuration and environment setup complete." << std::endl;

  // Read prompt from the favorite text editor,
  // like Git does when you write a commit message
  std::string prompt = get_prompt_from_editor();
  if (prompt.empty()) {
    throw std::runtime_error("Error: no prompt provided");
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
}
}  // namespace command

