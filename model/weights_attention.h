#ifndef WEIGHTS_ATTENTION_H
#define WEIGHTS_ATTENTION_H

#include <vector>
#include <string>

// Self-attention weights
extern std::vector<float> sa_qkv_weights;
//extern std::vector<float> sa_qkv_bias;
extern std::vector<float> sa_proj_weights;
//extern std::vector<float> sa_proj_bias;

// Cross-attention weights
extern std::vector<float> ca_qkv_weights;
//extern std::vector<float> ca_qkv_bias;
extern std::vector<float> ca_proj_weights;
//extern std::vector<float> ca_proj_bias;

// 1x1 convolution weights
extern std::vector<float> conv_weights;
extern std::vector<float> conv_bias;

// Configuration parameters
extern int attention_hidden_size;
extern int attention_num_heads;

void load_attention_weights(const std::string &path);

#endif // WEIGHTS_ATTENTION_H
