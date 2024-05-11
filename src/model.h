#pragma once

#include "model.h"
#include <torch/torch.h>
#include <cmath>
#include <vector>

namespace model {
class LayerNormImpl : public torch::nn::Module {
  torch::Tensor weight;
  torch::Tensor bias;
  bool use_bias;

 public:
  LayerNormImpl(int64_t ndim, bool bias);

  torch::Tensor forward(torch::Tensor input);
};

TORCH_MODULE(LayerNorm); // Wrapper to create shared_ptr<LayerNormImpl>

class CausalSelfAttentionImpl : public torch::nn::Module {
 private:
  torch::nn::Linear c_attn;
  torch::nn::Linear c_proj;
  torch::nn::Dropout attn_dropout;
  torch::nn::Dropout resid_dropout;
  int64_t n_head;
  int64_t n_embd;
  torch::Tensor attn_mask; // Attention mask for causal masking

 public:
  CausalSelfAttentionImpl(int64_t n_embd, int64_t n_head, double dropout, bool bias);

  torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(CausalSelfAttention);

class MLPImpl : public torch::nn::Module {
 private:
  torch::nn::Linear c_fc;   // First fully connected layer
  torch::nn::Linear c_proj; // Output projection layer
  torch::nn::Dropout dropout;

 public:
  MLPImpl(int64_t n_embd, double dropout, bool bias);

  torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(MLP); // Wrapper to create shared_ptr<MLPImpl>

class BlockImpl : public torch::nn::Module {
 private:
  LayerNorm ln_1; // First layer normalization
  LayerNorm ln_2; // Second layer normalization
  CausalSelfAttention attn; // Causal self-attention module
  MLP mlp; // Multi-layer perceptron module

 public:
  BlockImpl(int64_t n_embd, int64_t n_head, double dropout, bool bias);

  torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(Block); // Wrapper to create shared_ptr<BlockImpl>

struct Config {
  int64_t vocab_size;
  int64_t block_size;
  int64_t n_layer;
  int64_t n_head;
  int64_t n_embd;
  double dropout;
  bool bias;
};

class GPT : public torch::nn::Module {
 private:
  torch::nn::Embedding wte;
  torch::nn::Embedding wpe;
  torch::nn::Dropout drop;
  std::vector<torch::nn::AnyModule> h; // Use AnyModule to store different types of modules
  LayerNorm ln_f;
  torch::nn::Linear lm_head;

 public:
  GPT(const Config &config);

  std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor idx, torch::Tensor targets = {});
};
}
