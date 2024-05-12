#include "model.h"
#include <limits>
#include <cassert>

namespace model {
LayerNormImpl::LayerNormImpl(int64_t ndim, bool bias) : use_bias(bias) {
  weight = register_parameter("weight", torch::ones({ndim}));
  if (use_bias) {
    this->bias = register_parameter("bias", torch::zeros({ndim}));
  } else {
    this->bias = torch::Tensor(); // Create an empty tensor if bias is not used
  }
}

torch::Tensor LayerNormImpl::forward(torch::Tensor input) {
  if (use_bias) {
    return torch::layer_norm(input, {input.size(-1)}, weight, bias, 1e-5);
  } else {
    return torch::layer_norm(input, {input.size(-1)}, weight, torch::Tensor(), 1e-5);
  }
}

CausalSelfAttentionImpl::CausalSelfAttentionImpl(int64_t n_embd, int64_t n_head, double dropout, bool bias, int block_size, bool flash)
: c_attn(register_module("c_attn", torch::nn::Linear(torch::nn::LinearOptions(n_embd, 3 * n_embd).bias(bias)))), // key, query, value projections for all heads, but in a batch
  c_proj(register_module("c_proj", torch::nn::Linear(torch::nn::LinearOptions(n_embd, n_embd).bias(bias)))), // output projection
  attn_dropout(dropout), // regularization
  resid_dropout(dropout), // regularization
  n_head(n_head),
  n_embd(n_embd),
  dropout(dropout),
  flash(flash) {
    assert(n_embd % n_head == 0);
    if (!flash) {
      std::cout << "WARNING: using slow attention." << std::endl;
      // causal mask to ensure that attention is only applied to the left in the input sequence
      this->bias = register_buffer("bias", torch::tril(torch::ones({block_size, block_size}))
                                        .view({1, 1, block_size, block_size}));
   }
}


torch::Tensor CausalSelfAttentionImpl::forward(torch::Tensor x) {
  auto B = x.size(0); // Batch size
  auto T = x.size(1); // Sequence length
  auto C = x.size(2); // Embedding dimension

  // calculate query, key, values for all heads in batch and move head forward to be the batch dim
  auto tensors = c_attn(x).chunk(3, 2);
  auto k = tensors[0].view({B, T, n_head, C / n_head}).transpose(1, 2);  // (B, nh, T, hs)
  auto q = tensors[1].view({B, T, n_head, C / n_head}).transpose(1, 2);  // (B, nh, T, hs)
  auto v = tensors[2].view({B, T, n_head, C / n_head}).transpose(1, 2);  // (B, nh, T, hs)

  // causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
  torch::Tensor y;
  if (flash) {
    // efficient attention using Flash Attention CUDA kernels
    double scale = 1.0 / std::sqrt(k.size(-1));
    double dropout_p = this->is_training() ? this->dropout : 0.0;
    y = torch::native::scaled_dot_product_attention(q, k, v, std::nullopt, dropout_p, true, scale);
  } else {
    // manual implementation of attention
    auto att = (q.matmul(k.transpose(-2, -1))) * (1.0 / std::sqrt(k.size(-1)));
    att = att.masked_fill(bias.slice(0, 0, 0, T).slice(1, 0, T) == 0, -std::numeric_limits<float>::infinity());
    att = torch::softmax(att, -1);
    att = attn_dropout(att);
    y = att.matmul(v);  // (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
  }
  y = y.transpose(1, 2).contiguous().view({B, T, C});  // re-assemble all head outputs side by side

  // output projection
  return resid_dropout(c_proj(y));
}


MLPImpl::MLPImpl(int64_t n_embd, double dropout, bool bias)
    : c_fc(register_module("c_fc", torch::nn::Linear(torch::nn::LinearOptions(n_embd, 4 * n_embd).bias(bias)))),
      c_proj(register_module("c_proj", torch::nn::Linear(torch::nn::LinearOptions(4 * n_embd, n_embd).bias(bias)))),
      dropout(register_module("dropout", torch::nn::Dropout(dropout))) {}

torch::Tensor MLPImpl::forward(torch::Tensor x) {
  x = c_fc(x);
  x = torch::gelu(x);
  x = c_proj(x);
  x = dropout(x);
  return x;
}

BlockImpl::BlockImpl(int64_t n_embd, int64_t n_head, double dropout, bool bias, int block_size, bool flash)
    : ln_1(register_module("ln_1", LayerNorm(n_embd, bias))),
      ln_2(register_module("ln_2", LayerNorm(n_embd, bias))),
      attn(register_module("attn", CausalSelfAttention(n_embd, n_head, dropout, bias, block_size, flash))),
      mlp(register_module("mlp", MLP(n_embd, dropout, bias))) {}

torch::Tensor BlockImpl::forward(torch::Tensor x) {
  auto x_res = attn(ln_1->forward(x));
  x = x + x_res;
  x_res = mlp(ln_2->forward(x));
  x = x + x_res;
  return x;
}

GPT::GPT(const Config &config)
    : wte(torch::nn::Embedding(config.vocab_size, config.n_embd)),
      wpe(torch::nn::Embedding(config.block_size, config.n_embd)),
      drop(torch::nn::Dropout(config.dropout)),
      ln_f(LayerNorm(config.n_embd, config.bias)),
      lm_head(torch::nn::Linear(torch::nn::LinearOptions(config.n_embd, config.vocab_size).bias(false))) {
  register_module("wte", wte);
  register_module("wpe", wpe);
  register_module("drop", drop);
  register_module("ln_f", ln_f);
  register_module("lm_head", lm_head);
  for (int i = 0; i < config.n_layer; ++i) {
    auto block = Block(config.n_embd, config.n_head, config.dropout, config.bias, config.block_size, config.flash_attention);
    h.emplace_back(register_module("h_" + std::to_string(i), block));
  }
  // Tie weights
  lm_head->weight = wte->weight;
}

std::tuple<torch::Tensor, torch::Tensor> GPT::forward(torch::Tensor idx, torch::Tensor targets) {
  auto device = idx.device();
  auto pos = torch::arange(0, idx.size(1), device = device);
  auto tok_emb = wte(idx);
  auto pos_emb = wpe(pos);
  auto x = drop(tok_emb + pos_emb);

  for (auto &block : h) {
    x = block.forward(x);
  }

  x = ln_f(x);

  auto logits = lm_head(x);
  if (targets.defined()) {
    auto loss = torch::nn::functional::cross_entropy(logits.view({-1, logits.size(-1)}), targets.view(-1));
    return {logits, loss};
  } else {
    return {logits, {}};
  }
}
}
