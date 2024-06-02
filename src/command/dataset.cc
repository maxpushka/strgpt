#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

#include "command/command.h"
#include "curl/curl.h"
#include "tokenizer/bpe.h"
#include "tokenizer/char.h"
#include "tokenizer/tokenizer.h"

namespace command {
std::string download_data(const std::string& url) {
  CURL* curl;
  CURLcode res;
  std::string data;

  curl_global_init(CURL_GLOBAL_DEFAULT);
  curl = curl_easy_init();
  if (curl) {
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(
        curl, CURLOPT_WRITEFUNCTION,
        [](void* ptr, size_t size, size_t nmemb, void* userdata) -> size_t {
          std::string* data = static_cast<std::string*>(userdata);
          data->append(static_cast<char*>(ptr), size * nmemb);
          return size * nmemb;
        });
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &data);
    res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);
  }
  curl_global_cleanup();

  if (res != CURLE_OK) {
    throw std::runtime_error("Failed to download data: " +
                             std::string(curl_easy_strerror(res)));
  }

  return data;
}

void save_to_file(const std::string& path, const std::string& data) {
  std::ofstream ofs(path);
  if (!ofs) {
    throw std::runtime_error("Failed to open file: " + path);
  }
  ofs << data;
}

void save_bin_file(const std::string& path, const std::vector<int>& data) {
  std::ofstream ofs(path, std::ios::binary);
  if (!ofs) {
    throw std::runtime_error("Failed to open binary file: " + path);
  }
  ofs.write(reinterpret_cast<const char*>(data.data()),
            data.size() * sizeof(int));
}

// void save_meta_file(const std::string& path,
//                     const std::unordered_map<char, int>& stoi,
//                     const std::unordered_map<int, char>& itos) {
//   std::ofstream ofs(path, std::ios::binary);
//   if (!ofs) {
//     throw std::runtime_error("Failed to open meta file: " + path);
//   }
//   torch::save(stoi, ofs);
//   torch::save(itos, ofs);
// }

void prepare_data(const std::string& dataset_url, const std::string& out_path,
                  const tokenizer::Type tok_type) {
  namespace fs = std::filesystem;
  std::string input_file_path = fs::path(out_path) / "input.txt";

  // Download dataset if not already present
  if (!fs::exists(input_file_path)) {
    std::string data = download_data(dataset_url);
    save_to_file(input_file_path, data);
  }

  // Read the data
  std::ifstream ifs{input_file_path};
  std::string data((std::istreambuf_iterator<char>(ifs)),
                   std::istreambuf_iterator<char>());
  std::cout << "Length of dataset in characters: " << data.size() << std::endl;

  // Create the tokenizer
  std::unique_ptr<tokenizer::Tokenizer> tokenizer;
  switch (tok_type) {
    case tokenizer::Type::Char: {
      tokenizer = std::make_unique<tokenizer::CharLevel>(data);
      break;
    }
    case tokenizer::Type::BPE: {
      const char* tokenizer_config = std::getenv("TOKENIZER_CONFIG");
      if (tokenizer_config == nullptr) {
        throw std::runtime_error(
            "Error: environment variable TOKENIZER_CONFIG is not set");
      }

      std::stringstream config_path;
      config_path << tokenizer_config << "/tokenizer.json";
      std::ifstream config_file{config_path.str(), std::ios::in};

      tokenizer = std::make_unique<tokenizer::BPE>(config_path);
      break;
    }
    default:
      throw std::runtime_error("Error: tokenizer is not supported");
  };

  // Tokenize the data
  auto encoded_data = tokenizer->encode(data);
  std::cout << "Encoded data size: " << encoded_data.size() << std::endl;

  // Split the data into training and validation sets
  size_t n = encoded_data.size();
  std::vector<int> train_data(
      encoded_data.begin(),
      encoded_data.begin() + static_cast<size_t>(n * 0.9));
  std::vector<int> val_data(encoded_data.begin() + static_cast<size_t>(n * 0.9),
                            encoded_data.end());
  std::cout << "Train data size: " << train_data.size() << std::endl;
  std::cout << "Validation data size: " << val_data.size() << std::endl;

  // Save the train and validation data
  save_bin_file(fs::path(out_path) / "train.bin", train_data);
  save_bin_file(fs::path(out_path) / "val.bin", val_data);

  // Save the meta information
  // save_meta_file(fs::path(out_path) / "meta.bin", tokenizer.stoi_,
  //                tokenizer.itos_);
}
}  // namespace command
