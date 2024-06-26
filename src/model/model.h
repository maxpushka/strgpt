#pragma once

#include <cmath>
#include <memory>
#include <tuple>
#include <utility>

#include "nlohmann/json.hpp"
#include "torch/torch.h"

namespace model {
struct Config {
  int64_t vocab_size = 1024;
  int64_t block_size = 50304;  // GPT-2 vocab_size of 50257, padded up to
                               // nearest multiple of 64 for efficiency
  int64_t n_layer = 12;
  int64_t n_head = 12;
  int64_t n_embd = 768;
  double dropout = 0.0;
  bool bias = true;  // true: bias in Linears and LayerNorms, like GPT-2. false:
                     // a bit better and faster
  bool flash_attention = true;

  NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(Config, vocab_size, block_size,
                                              n_layer, n_head, n_embd, dropout,
                                              bias, flash_attention)
};

class LayerNormImpl : public torch::nn::Module {
  torch::Tensor weight;
  torch::Tensor bias;
  bool use_bias;

 public:
  LayerNormImpl(int64_t ndim, bool bias);

  torch::Tensor forward(torch::Tensor input);
};

TORCH_MODULE(LayerNorm);  // Wrapper to create shared_ptr<LayerNormImpl>

class CausalSelfAttentionImpl : public torch::nn::Module {
 private:
  torch::nn::Linear c_attn;
  torch::nn::Linear c_proj;
  torch::nn::Dropout attn_dropout;
  torch::nn::Dropout resid_dropout;
  int64_t n_head;
  int64_t n_embd;
  float dropout;
  bool flash;
  torch::Tensor bias;

 public:
  explicit CausalSelfAttentionImpl(const Config& config);

  torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(CausalSelfAttention);

class MLPImpl : public torch::nn::Module {
 private:
  torch::nn::Linear c_fc;    // First fully connected layer
  torch::nn::Linear c_proj;  // Output projection layer
  torch::nn::Dropout dropout;

 public:
  MLPImpl(int64_t n_embd, double dropout, bool bias);

  torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(MLP);  // Wrapper to create shared_ptr<MLPImpl>

class BlockImpl : public torch::nn::Module {
 private:
  LayerNorm ln_1;            // First layer normalization
  LayerNorm ln_2;            // Second layer normalization
  CausalSelfAttention attn;  // Causal self-attention module
  MLP mlp;                   // Multi-layer perceptron module

 public:
  explicit BlockImpl(const Config& config);

  torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(Block);  // Wrapper to create shared_ptr<BlockImpl>

struct TransformerImpl : public torch::nn::Module {
  torch::nn::Embedding wte;
  torch::nn::Embedding wpe;
  torch::nn::Dropout drop;
  torch::nn::ModuleList h;
  LayerNorm ln_f;

  explicit TransformerImpl(const Config& config);
};

TORCH_MODULE(Transformer);  // Wrapper to create shared_ptr<BlockImpl>

class GPT : public torch::nn::Module {
  Config config;
  Transformer transformer;
  torch::nn::Linear lm_head;

 public:
  explicit GPT(const Config& config);

  void init_weights(torch::nn::Module* module);

  std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& idx,
      const torch::optional<torch::Tensor>& targets = {});

  int64_t get_num_params(bool non_embedding = true) const;
  void crop_block_size(int64_t block_size);
  std::shared_ptr<torch::optim::Optimizer> configure_optimizers(
      double weight_decay, double learning_rate,
      std::tuple<double, double> betas, const torch::Device& device_type);
  double estimate_mfu(int64_t fwdbwd_per_iter, double dt) const;
  torch::Tensor generate(torch::Tensor idx, int64_t max_new_tokens,
                         double temperature = 1.0, int64_t top_k = -1);
};
}  // namespace model
