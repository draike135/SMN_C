#include "weights_attention.h"
#include "utils.h"

// Self-attention weights
std::vector<float> sa_qkv_weights;
std::vector<float> sa_qkv_bias;
std::vector<float> sa_proj_weights;
std::vector<float> sa_proj_bias;

// Cross-attention weights
std::vector<float> ca_qkv_weights;
std::vector<float> ca_qkv_bias;
std::vector<float> ca_proj_weights;
std::vector<float> ca_proj_bias;

// 1x1 convolution weights
std::vector<float> conv_weights;
std::vector<float> conv_bias;

// Configuration parameters
int attention_hidden_size = 256;
int attention_num_heads = 8;

void load_attention_weights(const std::string &path) {
    // Load self-attention weights
    sa_qkv_weights = load_binary_file(path + "/sa_qkv_weights.bin");
    //sa_qkv_bias = load_binary_file(path + "/sa_qkv_bias.bin");
    sa_proj_weights = load_binary_file(path + "/sa_proj_weights.bin");
    //sa_proj_bias = load_binary_file(path + "/sa_proj_bias.bin");

    // Load cross-attention weights
    ca_qkv_weights = load_binary_file(path + "/ca_qkv_weights.bin");
    //ca_qkv_bias = load_binary_file(path + "/ca_qkv_bias.bin");
    ca_proj_weights = load_binary_file(path + "/ca_proj_weights.bin");
    //ca_proj_bias = load_binary_file(path + "/ca_proj_bias.bin");

    // Load 1x1 convolution weights
    conv_weights = load_binary_file(path + "/conv_weights.bin");
    conv_bias = load_binary_file(path + "/conv_bias.bin");
}
