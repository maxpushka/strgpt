#pragma once

#include <string>

#include "model.h"

// Configuration struct to hold all training parameters
struct Config {
  std::string data_dir = "/home/maxpushka/dev/github.com/maxpushka/strgpt/data/shakespeare_char";
  std::string out_dir = "/home/maxpushka/dev/github.com/maxpushka/strgpt/out";

  int eval_interval = 2000;
  int log_interval = 1;
  int eval_iters = 200;
  bool eval_only = false;
  bool always_save_checkpoint = true;
  std::string init_from = "resume"; // 'scratch' or 'resume' or 'gpt2*'
  std::string dataset = "openwebtext";
  int gradient_accumulation_steps = 40; // 5 * 8
  int batch_size = 32;
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

size_t load_checkpoint(const std::string& path, std::shared_ptr<GPT> model, std::shared_ptr<torch::optim::Optimizer> optimizer);

void train_model_with_scheduler_and_checkpointing(std::shared_ptr<GPT> model,
                                                  std::shared_ptr<torch::optim::Optimizer> optimizer,
                                                  const Config &cfg,
                                                  size_t prev_iters_count,
                                                  torch::Device device);
