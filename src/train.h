#pragma once

#include <string>
#include <nlohmann/json.hpp>

#include "model.h"

namespace train {
// Configuration struct to hold all training parameters.
// Default config values designed to train a gpt2 (124M) on OpenWebText.
struct Config {
  // I/O
  std::string data_dir;
  std::string out_dir = "out";
  int eval_interval = 2000;
  int log_interval = 1;
  int eval_iters = 200;
  bool eval_only = false; // if true, the program exits right after the first eval
  bool always_save_checkpoint = true; // if true, always save a checkpoint after each eval
  std::string init_from = "scratch"; // 'scratch' or 'resume' or 'gpt2*'
  // WandB logging
  // wandb_log = False # disabled by default
  // wandb_project = 'owt'
  // wandb_run_name = 'gpt2' # 'run' + str(time.time())
  // Data
  std::string dataset = "openwebtext";
  int gradient_accumulation_steps = 5*8; // used to simulate larger batch sizes
  int batch_size = 12; // if gradient_accumulation_steps > 1, this is the micro-batch size
  // Model
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
  // DDP settings
  // std::string backend = "nccl"; // 'nccl', 'gloo', etc.
  // System
  std::string device = "cuda"; // examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
  std::string dtype = "float16"; // 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
  bool compile = true; // compile the model to be faster

  NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(Config, data_dir, out_dir, eval_interval, log_interval, eval_iters, eval_only,
                                              always_save_checkpoint, init_from, dataset, gradient_accumulation_steps, batch_size,
                                              model, learning_rate, max_iters, weight_decay, beta1, beta2, grad_clip, decay_lr,
                                              warmup_iters, lr_decay_iters, min_lr, device, dtype, compile)
};

size_t load_checkpoint(const std::string& path, std::shared_ptr<model::GPT> model, std::shared_ptr<torch::optim::Optimizer> optimizer);

void train_model_with_scheduler_and_checkpointing(std::shared_ptr<model::GPT> model,
                                                  std::shared_ptr<torch::optim::Optimizer> optimizer,
                                                  const Config &cfg,
                                                  size_t prev_iters_count,
                                                  torch::Device device);
}
