#include "model.h"
#include <limits>

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

CausalSelfAttentionImpl::CausalSelfAttentionImpl(int64_t n_embd, int64_t n_head, double dropout, bool bias)
    : c_attn(register_module("c_attn", torch::nn::Linear(torch::nn::LinearOptions(n_embd, 3 * n_embd).bias(bias)))),
      c_proj(register_module("c_proj", torch::nn::Linear(torch::nn::LinearOptions(n_embd, n_embd).bias(bias)))),
      attn_dropout(register_module("attn_dropout", torch::nn::Dropout(dropout))),
      resid_dropout(register_module("resid_dropout", torch::nn::Dropout(dropout))),
      n_head(n_head), n_embd(n_embd) {
  // Initialize bias tensor for masked attention; should be 1 x 1 x T x T where T is the sequence length.
  // T will be dynamically set during the forward pass.
}

torch::Tensor CausalSelfAttentionImpl::forward(torch::Tensor x) {
  auto B = x.size(0); // Batch size
  auto T = x.size(1); // Sequence length
  auto C = x.size(2); // Embedding dimension

  // Ensure that the attention mask is correctly sized each forward pass
  attn_mask = torch::tril(torch::ones({1, 1, T, T}, x.options())).contiguous();

  // Split the concatenated tensor to get query, key and values
  auto tensors = c_attn(x).chunk(3, 2);
  auto q = tensors[0];
  auto k = tensors[1];
  auto v = tensors[2];

  // Reshape and transpose tensors for multi-head attention
  auto shape = torch::IntArrayRef({B, T, n_head, C / n_head});
  k = k.view(shape).transpose(1, 2);
  q = q.view(shape).transpose(1, 2);
  v = v.view(shape).transpose(1, 2);

  // Compute scaled dot-product attention
  auto attn = torch::matmul(q, k.transpose(-2, -1)) * (1.0 / std::sqrt(C / n_head));
  attn = attn.masked_fill(attn_mask.to(attn.device()) == 0, -std::numeric_limits<float>::infinity());
  attn = torch::softmax(attn, -1);
  attn = attn_dropout(attn);
  auto y = torch::matmul(attn, v);

  // Reassemble all head outputs
  y = y.transpose(1, 2).contiguous().view({B, T, C});

  // Apply output projection and dropout
  y = resid_dropout(c_proj(y));
  return y;
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

BlockImpl::BlockImpl(int64_t n_embd, int64_t n_head, double dropout, bool bias)
    : ln_1(register_module("ln_1", LayerNorm(n_embd, bias))),
      ln_2(register_module("ln_2", LayerNorm(n_embd, bias))),
      attn(register_module("attn", CausalSelfAttention(n_embd, n_head, dropout, bias))),
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
    auto block = Block(config.n_embd, config.n_head, config.dropout, config.bias);
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
