#pragma once

#include <string>
#include <nlohmann/json.hpp>

#include "model.h"

namespace train {
struct DataConfig {
  std::string data_dir;
  std::string dataset = "openwebtext";
  std::string train_file = "train.bin";
  std::string val_file = "val.bin";

  NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(DataConfig, data_dir, dataset, train_file, val_file);
};

struct TrainConfig {
  std::string out_dir = "out";
  int eval_interval = 2000;
  int eval_iters = 200;
  bool eval_only = false; // if true, the program exits right after the first eval
  int log_interval = 1;
  bool always_save_checkpoint = true; // if true, always save a checkpoint after each eval
  std::string init_from = "scratch"; // 'scratch' or 'resume' or 'gpt2*'
  int gradient_accumulation_steps = 5*8; // used to simulate larger batch sizes
  int batch_size = 12; // if gradient_accumulation_steps > 1, this is the micro-batch size
  // AdamW optimizer
  float learning_rate = 6e-4; // max learning rate
  int max_iters = 600000; // total number of training iterations
  float weight_decay = 0.1;
  float beta1 = 0.9;
  float beta2 = 0.95;
  float grad_clip = 1.0; // clip gradients at this value, or disable if == 0.0
  // Learning rate decay settings
  bool decay_lr = true; // whether to decay the learning rate
  int warmup_iters = 2000; // how many steps to warm up for
  int lr_decay_iters = 600000; // should be ~= max_iters
  float min_lr = 6e-5; // minimum learning rate, should be ~= learning_rate/10
  // System
  std::string device = "cuda"; // examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
  std::string dtype = "float16"; // 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler

  NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(TrainConfig, out_dir, eval_interval, eval_iters, eval_only, log_interval, always_save_checkpoint, init_from, gradient_accumulation_steps,
                                              batch_size, learning_rate, max_iters, weight_decay, beta1, beta2, grad_clip, decay_lr, warmup_iters, lr_decay_iters, min_lr, device, dtype 
  );
};

// Configuration struct to hold all training parameters.
// Default config values designed to train a gpt2 (124M) on OpenWebText.
struct Config {
  DataConfig data;
  TrainConfig train;
  model::Config model{
      .vocab_size = 50257,
      .block_size = 1024, // context of up to 256 previous characters
      .n_layer = 12,
      .n_head = 12,
      .n_embd = 768,
      .dropout = 0.0,
      .bias = false,
      .flash_attention = true
  };

  NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(Config, data, train, model)
};

void train_model(std::shared_ptr<model::GPT> model,
                 std::shared_ptr<torch::optim::Optimizer> optimizer,
                 const Config &cfg,
                 torch::Device device);
}
