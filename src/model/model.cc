#include "model/model.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace model {
LayerNormImpl::LayerNormImpl(int64_t ndim, bool bias) : use_bias(bias) {
  weight = register_parameter("weight", torch::ones({ndim}));
  if (use_bias) {
    this->bias = register_parameter("bias", torch::zeros({ndim}));
  } else {
    this->bias =
        register_parameter("bias", torch::Tensor(), /*requires_grad=*/false);
  }
}

torch::Tensor LayerNormImpl::forward(torch::Tensor input) {
  return torch::layer_norm(input, {input.size(-1)}, weight, bias, 1e-5);
}

CausalSelfAttentionImpl::CausalSelfAttentionImpl(const Config &config)
    : c_attn(register_module(
          "c_attn",
          torch::nn::Linear(
              torch::nn::LinearOptions(config.n_embd,
                                       3 * config.n_embd)
                  .bias(config.bias)))),  // key, query, value projections for
                                          // all heads, but in a batch
      c_proj(register_module(
          "c_proj",
          torch::nn::Linear(torch::nn::LinearOptions(config.n_embd,
                                                     config.n_embd)
                                .bias(config.bias)))),  // output projection
      attn_dropout(config.dropout),                     // regularization
      resid_dropout(config.dropout),                    // regularization
      n_head(config.n_head),
      n_embd(config.n_embd),
      dropout(config.dropout),
      flash(config.flash_attention) {
  assert(config.n_embd % config.n_head == 0);
  if (!flash) {
    std::cout << "WARNING: using slow attention." << std::endl;
    // causal mask to ensure that attention is only applied to the left in the
    // input sequence
    this->bias = register_buffer(
        "bias", torch::tril(torch::ones({config.block_size, config.block_size}))
                    .view({1, 1, config.block_size, config.block_size}));
  }
}

torch::Tensor CausalSelfAttentionImpl::forward(torch::Tensor x) {
  auto B = x.size(0);  // Batch size
  auto T = x.size(1);  // Sequence length
  auto C = x.size(2);  // Embedding dimension

  // calculate query, key, values for all heads in batch and move head forward
  // to be the batch dim
  auto tensors = c_attn(x).chunk(3, 2);
  auto k = tensors[0]
               .view({B, T, n_head, C / n_head})
               .transpose(1, 2);  // (B, nh, T, hs)
  auto q = tensors[1]
               .view({B, T, n_head, C / n_head})
               .transpose(1, 2);  // (B, nh, T, hs)
  auto v = tensors[2]
               .view({B, T, n_head, C / n_head})
               .transpose(1, 2);  // (B, nh, T, hs)

  // causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B,
  // nh, T, T)
  torch::Tensor y;
  if (flash) {
    // efficient attention using Flash Attention CUDA kernels
    double scale = 1.0 / std::sqrt(k.size(-1));
    double dropout_p = this->is_training() ? this->dropout : 0.0;
    y = torch::native::scaled_dot_product_attention(q, k, v, std::nullopt,
                                                    dropout_p, true, scale);
  } else {
    // manual implementation of attention
    namespace I = torch::indexing;
    auto att = (q.matmul(k.transpose(-2, -1))) * (1.0 / std::sqrt(k.size(-1)));
    att = att.masked_fill(
        bias.index({I::Slice(), I::Slice(), I::Slice(I::None, T),
                    I::Slice(I::None, T)}) == 0,
        -std::numeric_limits<float>::infinity());
    att = torch::softmax(att, -1);
    att = attn_dropout(att);
    y = att.matmul(v);  // (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
  }
  y = y.transpose(1, 2).contiguous().view(
      {B, T, C});  // re-assemble all head outputs side by side

  // output projection
  return resid_dropout(c_proj(y));
}

MLPImpl::MLPImpl(int64_t n_embd, double dropout, bool bias)
    : c_fc(register_module(
          "c_fc",
          torch::nn::Linear(
              torch::nn::LinearOptions(n_embd, 4 * n_embd).bias(bias)))),
      c_proj(register_module(
          "c_proj",
          torch::nn::Linear(
              torch::nn::LinearOptions(4 * n_embd, n_embd).bias(bias)))),
      dropout(register_module("dropout", torch::nn::Dropout(dropout))) {}

torch::Tensor MLPImpl::forward(torch::Tensor x) {
  x = c_fc(x);
  x = torch::gelu(x);
  x = c_proj(x);
  x = dropout(x);
  return x;
}

BlockImpl::BlockImpl(const Config &config)
    : ln_1(register_module("ln_1", LayerNorm(config.n_embd, config.bias))),
      ln_2(register_module("ln_2", LayerNorm(config.n_embd, config.bias))),
      attn(register_module("attn", CausalSelfAttention(config))),
      mlp(register_module("mlp",
                          MLP(config.n_embd, config.dropout, config.bias))) {}

torch::Tensor BlockImpl::forward(torch::Tensor x) {
  auto x_res = attn(ln_1->forward(x));
  x = x + x_res;
  x_res = mlp(ln_2->forward(x));
  x = x + x_res;
  return x;
}

TransformerImpl::TransformerImpl(const Config &config)
    : wte(register_module(
          "wte", torch::nn::Embedding(config.vocab_size, config.n_embd))),
      wpe(register_module(
          "wpe", torch::nn::Embedding(config.block_size, config.n_embd))),
      drop(register_module("drop", torch::nn::Dropout(config.dropout))),
      h(register_module("h", torch::nn::ModuleList())),
      ln_f(register_module("ln_f", LayerNorm(config.n_embd, config.bias))) {
  // Initialize the vector of blocks
  for (int i = 0; i < config.n_layer; ++i) {
    h->push_back(Block(config));
  }
}

GPT::GPT(const Config &config)
    : config(config),
      transformer(register_module("transformer", Transformer(config))),
      lm_head(register_module(
          "lm_head", torch::nn::Linear(torch::nn::LinearOptions(
                                           config.n_embd, config.vocab_size)
                                           .bias(false)))) {
  assert(config.vocab_size > 0);
  assert(config.block_size > 0);

  transformer->wte->weight =
      lm_head->weight;  // https://paperswithcode.com/method/weight-tying

  // init all weights
  apply([this](torch::nn::Module &module) { this->init_weights(&module); });

  // apply special scaled init to the residual projections, per GPT-2 paper
  for (auto &param : named_parameters()) {
    if (param.key().ends_with("c_proj.weight")) {
      torch::nn::init::normal_(param.value(), 0.0,
                               0.02 / std::sqrt(2.0 * config.n_layer));
    }
  }

  // report number of parameters
  std::cout << "Number of parameters: " << get_num_params() / 1e6 << "M\n";
}

void GPT::init_weights(torch::nn::Module *module) {
  if (auto *linear = dynamic_cast<torch::nn::LinearImpl *>(module)) {
    torch::nn::init::normal_(linear->weight, 0.0, 0.02);
    if (linear->options.bias()) {
      torch::nn::init::zeros_(linear->bias);
    }
  } else if (auto *embedding =
                 dynamic_cast<torch::nn::EmbeddingImpl *>(module)) {
    torch::nn::init::normal_(embedding->weight, 0.0, 0.02);
  }
}

// Return the number of parameters in the model.
// For non-embedding count (default), the position embeddings get subtracted.
// The token embeddings would too, except due to the parameter sharing these
// params are actually used as weights in the final layer, so we include them.
int64_t GPT::get_num_params(bool non_embedding) const {
  int64_t n_params = 0;
  for (const auto &param : parameters()) {
    n_params += param.numel();
  }
  if (non_embedding) {
    n_params -= transformer->wpe->weight.numel();
  }
  return n_params;
}

std::tuple<torch::Tensor, torch::Tensor> GPT::forward(
    const torch::Tensor &idx, const torch::optional<torch::Tensor> &targets) {
  auto device = idx.device();
  auto T = idx.sizes()[1];
  assert(T <= config.block_size);

  auto pos = torch::arange(
      T,
      torch::TensorOptions().dtype(torch::kLong).device(device));  // shape (t)

  auto tok_emb = transformer->wte->forward(
      idx);  // token embeddings of shape (b, t, n_embd)
  auto pos_emb = transformer->wpe->forward(
      pos);  // position embeddings of shape (t, n_embd)
  auto x = transformer->drop->forward(tok_emb + pos_emb);

  for (const auto &block : *transformer->h) {
    x = block->as<Block>()->forward(x);
  }

  x = transformer->ln_f->forward(x);

  torch::Tensor logits;
  torch::Tensor loss;
  if (targets.has_value()) {
    // if we are given some desired targets also calculate the loss
    namespace F = torch::nn::functional;
    logits = lm_head->forward(x);
    loss = F::cross_entropy(logits.view({-1, logits.size(-1)}),
                            targets.value().view(-1),
                            F::CrossEntropyFuncOptions().ignore_index(-1));
  } else {
    // inference-time mini-optimization: only forward the lm_head on the very
    // last position
    namespace I = torch::indexing;
    logits = lm_head->forward(x.index({I::Slice(), {-1}, I::Slice()}));
    loss = torch::Tensor();
  }

  return std::make_tuple(logits, loss);
}

// Model surgery to decrease the block size if necessary
// e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
// but want to use a smaller block size for some smaller, simpler model
void GPT::crop_block_size(int64_t block_size) {
  assert(block_size <= config.block_size);
  config.block_size = block_size;

  // Resize the positional embeddings
  namespace I = torch::indexing;
  transformer->wpe->weight =
      transformer->wpe->weight.index({I::Slice(I::None, block_size)});

  // Resize attention biases in all blocks if they exist
  for (auto &block : *transformer->h) {
    auto bias = block->named_parameters().find("attn.bias");
    if (bias == nullptr) continue;
    *bias = bias->index({I::Slice(), I::Slice(), I::Slice(I::None, block_size),
                         I::Slice(I::None, block_size)});
  }
}

/*
static std::shared_ptr<GPT> from_pretrained(const std::string& model_type, const
std::optional<std::unordered_map<std::string, double>>& override_args =
std::nullopt) { assert(model_type == "gpt2" || model_type == "gpt2-medium" ||
model_type == "gpt2-large" || model_type == "gpt2-xl");

    std::unordered_map<std::string, int64_t> config_args = {
        {"gpt2", {12, 12, 768}},
        {"gpt2-medium", {24, 16, 1024}},
        {"gpt2-large", {36, 20, 1280}},
        {"gpt2-xl", {48, 25, 1600}}
    }[model_type];

    std::cout << "Forcing vocab_size=50257, block_size=1024, bias=True\n";
    config_args["vocab_size"] = 50257;
    config_args["block_size"] = 1024;
    config_args["bias"] = true;

    if (override_args.has_value()) {
        if (override_args.value().count("dropout")) {
            std::cout << "Overriding dropout rate to " <<
override_args.value().at("dropout") << "\n"; config_args["dropout"] =
override_args.value().at("dropout");
        }
    }

    Config config{config_args["block_size"], config_args["vocab_size"],
config_args["n_layer"], config_args["n_head"], config_args["n_embd"],
config_args["dropout"], config_args["bias"]}; auto model =
std::make_shared<GPT>(config);

    auto sd = model->named_parameters();
    auto sd_keys = sd.keys();

    GPT2LMHeadModel model_hf = GPT2LMHeadModel::from_pretrained(model_type);
    auto sd_hf = model_hf.state_dict();

    for (const auto& k : sd_keys) {
        if (sd_hf.count(k)) {
            if (std::find(transposed.begin(), transposed.end(), k) !=
transposed.end()) { sd[k].copy_(sd_hf[k].t()); } else { sd[k].copy_(sd_hf[k]);
            }
        }
    }

    return model;
}
*/

std::shared_ptr<torch::optim::Optimizer> GPT::configure_optimizers(
    double weight_decay, double learning_rate, std::tuple<double, double> betas,
    const torch::Device &device) {
  // Start with all of the candidate parameters
  std::unordered_map<std::string, torch::Tensor> param_dict;
  for (const auto &param : this->named_parameters()) {
    if (param.value().requires_grad()) {
      param_dict[param.key()] = param.value();
    }
  }

  // Create optim groups. Any parameters that is 2D will be weight decayed,
  // otherwise no. i.e. all weight tensors in matmuls + embeddings decay, all
  // biases and layernorms don't.
  std::vector<torch::Tensor> decay_params;
  std::vector<torch::Tensor> nodecay_params;
  for (const auto &param : param_dict) {
    if (param.second.dim() >= 2) {
      decay_params.push_back(param.second);
    } else {
      nodecay_params.push_back(param.second);
    }
  }

  std::vector<torch::optim::OptimizerParamGroup> optim_groups;
  optim_groups.emplace_back(
      decay_params,
      std::make_unique<torch::optim::AdamWOptions>(
          torch::optim::AdamWOptions().weight_decay(weight_decay)));
  optim_groups.emplace_back(
      nodecay_params, std::make_unique<torch::optim::AdamWOptions>(
                          torch::optim::AdamWOptions().weight_decay(0.0)));

  int64_t num_decay_params = 0;
  for (const auto &p : decay_params) {
    num_decay_params += p.numel();
  }
  int64_t num_nodecay_params = 0;
  for (const auto &p : nodecay_params) {
    num_nodecay_params += p.numel();
  }

  std::cout << "num decayed parameter tensors: " << decay_params.size()
            << ", with " << num_decay_params << " parameters" << std::endl;
  std::cout << "num non-decayed parameter tensors: " << nodecay_params.size()
            << ", with " << num_nodecay_params << " parameters" << std::endl;

  // Create AdamW optimizer and use the fused version if it is available
  auto optim_defaults =
      torch::optim::AdamWOptions(learning_rate)
          .betas(betas);  // TODO: use fused version on CUDA patforms
  std::cout << "using fused AdamW: fused = " << std::boolalpha << false
            << std::endl;
  return std::make_shared<torch::optim::AdamW>(optim_groups, optim_defaults);
}

// Estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS
double GPT::estimate_mfu(int64_t fwdbwd_per_iter, double dt) const {
  // first estimate the number of flops we do per iteration.
  // see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
  int64_t N = get_num_params();
  auto L = config.n_layer;
  auto H = config.n_head;
  auto Q = config.n_embd / config.n_head;
  auto T = config.block_size;

  auto flops_per_token = 6 * N + 12 * L * H * Q * T;
  auto flops_per_fwdbwd = flops_per_token * T;
  auto flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter;
  // express our flops throughput as ratio of A100 bfloat16 peak flops
  auto flops_achieved = flops_per_iter / dt;  // per second
  auto flops_promised = 312e12;  // A100 GPU bfloat16 peak flops is 312 TFLOPS

  auto mfu = flops_achieved / flops_promised;
  return mfu;
}

// Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and
// complete the sequence max_new_tokens times, feeding the predictions back into
// the model each time. Most likely you'll want to make sure to be in
// model.eval() mode of operation for this.
torch::Tensor GPT::generate(torch::Tensor idx, int64_t max_new_tokens,
                            double temperature, int64_t top_k) {
  torch::NoGradGuard no_grad;
  auto block_size = config.block_size;

  for (int64_t i = 0; i < max_new_tokens; ++i) {
    // If the sequence context is growing too long, we must crop it at
    // block_size
    auto idx_cond = idx.size(1) <= block_size
                        ? idx
                        : idx.slice(1, idx.size(1) - block_size, idx.size(1));

    // Forward the model to get the logits for the index in the sequence
    auto [logits, _] = forward(idx_cond);

    // Pluck the logits at the final step and scale by desired temperature
    logits = logits.slice(1, logits.size(1) - 1, logits.size(1)) / temperature;

    // Optionally crop the logits to only the top k options
    if (top_k > 0) {
      top_k = std::min(top_k, logits.size(-1));
      auto top_k_values = std::get<0>(torch::topk(logits, top_k, -1));
      auto min_top_k_value = top_k_values.slice(-1, top_k - 1, top_k);
      logits = torch::where(
          logits < min_top_k_value,
          torch::full_like(logits, -std::numeric_limits<float>::infinity()),
          logits);
    }

    // Apply softmax to convert logits to (normalized) probabilities
    auto probs = torch::softmax(logits, -1);

    // Sample from the distribution
    auto idx_next = torch::multinomial(probs, 1);

    // Append sampled index to the running sequence and continue
    idx = torch::cat({idx, idx_next}, 1);
  }

  return idx;
}
}  // namespace model
