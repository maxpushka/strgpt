#include <torch/torch.h>
#include <cmath>
#include <vector>
#include <limits>

class LayerNormImpl : public torch::nn::Module {
    torch::Tensor weight;
    torch::Tensor bias;
    bool use_bias;

public:
    LayerNormImpl(int64_t ndim, bool bias) : use_bias(bias) {
        weight = register_parameter("weight", torch::ones({ndim}));
        if (use_bias) {
            this->bias = register_parameter("bias", torch::zeros({ndim}));
        } else {
            this->bias = torch::Tensor(); // Create an empty tensor if bias is not used
        }
    }

    torch::Tensor forward(torch::Tensor input) {
        if (use_bias) {
            return torch::layer_norm(input, {input.size(-1)}, weight, bias, 1e-5);
        } else {
            return torch::layer_norm(input, {input.size(-1)}, weight, torch::Tensor(), 1e-5);
        }
    }
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
    CausalSelfAttentionImpl(int64_t n_embd, int64_t n_head, double dropout, bool bias)
    : c_attn(register_module("c_attn", torch::nn::Linear(torch::nn::LinearOptions(n_embd, 3 * n_embd).bias(bias)))),
      c_proj(register_module("c_proj", torch::nn::Linear(torch::nn::LinearOptions(n_embd, n_embd).bias(bias)))),
      attn_dropout(register_module("attn_dropout", torch::nn::Dropout(dropout))),
      resid_dropout(register_module("resid_dropout", torch::nn::Dropout(dropout))),
      n_head(n_head), n_embd(n_embd) {
        // Initialize bias tensor for masked attention
        attn_mask = register_buffer("attn_mask", torch::tril(torch::ones({n_head, n_head})).unsqueeze(0));
    }

    torch::Tensor forward(torch::Tensor x) {
        auto B = x.size(0); // Batch size
        auto T = x.size(1); // Sequence length
        auto C = x.size(2); // Embedding dimension

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
};

TORCH_MODULE(CausalSelfAttention);

class MLPImpl : public torch::nn::Module {
private:
    torch::nn::Linear c_fc;   // First fully connected layer
    torch::nn::Linear c_proj; // Output projection layer
    torch::nn::Dropout dropout;

public:
    MLPImpl(int64_t n_embd, double dropout, bool bias)
    : c_fc(register_module("c_fc", torch::nn::Linear(torch::nn::LinearOptions(n_embd, 4 * n_embd).bias(bias)))),
      c_proj(register_module("c_proj", torch::nn::Linear(torch::nn::LinearOptions(4 * n_embd, n_embd).bias(bias)))),
      dropout(register_module("dropout", torch::nn::Dropout(dropout))) {}

    torch::Tensor forward(torch::Tensor x) {
        x = c_fc(x);
        x = torch::gelu(x);
        x = c_proj(x);
        x = dropout(x);
        return x;
    }
};

TORCH_MODULE(MLP); // Wrapper to create shared_ptr<MLPImpl>

class BlockImpl : public torch::nn::Module {
private:
    LayerNorm ln_1; // First layer normalization
    LayerNorm ln_2; // Second layer normalization
    CausalSelfAttention attn; // Causal self-attention module
    MLP mlp; // Multi-layer perceptron module

public:
    BlockImpl(int64_t n_embd, int64_t n_head, double dropout, bool bias)
    : ln_1(register_module("ln_1", LayerNorm(n_embd, bias))),
      ln_2(register_module("ln_2", LayerNorm(n_embd, bias))),
      attn(register_module("attn", CausalSelfAttention(n_embd, n_head, dropout, bias))),
      mlp(register_module("mlp", MLP(n_embd, dropout, bias))) {}

    torch::Tensor forward(torch::Tensor x) {
        auto x_res = attn(ln_1->forward(x));
        x = x + x_res;
        x_res = mlp(ln_2->forward(x));
        x = x + x_res;
        return x;
    }
};

TORCH_MODULE(Block); // Wrapper to create shared_ptr<BlockImpl>

struct GPTConfig {
    int64_t vocab_size;
    int64_t block_size;
    int64_t n_layer;
    int64_t n_head;
    int64_t n_embd;
    double dropout;
    bool bias;
};

class GPTImpl : public torch::nn::Module {
private:
    torch::nn::Embedding wte;
    torch::nn::Embedding wpe;
    torch::nn::Dropout drop;
    std::vector<torch::nn::AnyModule> h; // Use AnyModule to store different types of modules
    LayerNorm ln_f;
    torch::nn::Linear lm_head;

public:
    GPTImpl(const GPTConfig& config)
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

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor idx, torch::Tensor targets = {}) {
        auto device = idx.device();
        auto pos = torch::arange(0, idx.size(1), device=device);
        auto tok_emb = wte(idx);
        auto pos_emb = wpe(pos);
        auto x = drop(tok_emb + pos_emb);

        for (auto& block : h) {
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
};

TORCH_MODULE(GPT); // Wrapper to create shared_ptr<GPTImpl>
