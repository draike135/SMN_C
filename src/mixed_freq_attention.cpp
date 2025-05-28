#include "mixed_freq_attention.h"
#include "weights_attention.h"
#include <cmath>

MixedFrequencyAttentionParams create_mixed_freq_attention_params(
    int hidden_size,
    int num_heads) {

    MixedFrequencyAttentionParams params;

    int half_hidden = hidden_size / 2;
    int head_dim = half_hidden / num_heads;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Self-attention parameters (operates on half_hidden dimensions)
    params.sa_params = {
        half_hidden,     // dim
        num_heads,       // num_heads
        head_dim,        // head_dim
        scale,           // scale
        sa_qkv_weights,  // qkv_weights
        std::vector<float>(),     // qkv_bias
        sa_proj_weights, // proj_weights
        std::vector<float>()     // proj_bias
    };

    // Cross-attention parameters (operates on half_hidden dimensions)
    params.ca_params = {
        half_hidden,     // dim
        num_heads,       // num_heads
        head_dim,        // head_dim
        scale,           // scale
        ca_qkv_weights,  // qkv_weights
        std::vector<float>(),     // qkv_bias
        ca_proj_weights, // proj_weights
        std::vector<float>()     // proj_bias
    };

    // 1x1 convolution parameters
    params.conv_params = {
        hidden_size,    // in_channels
        hidden_size,    // out_channels
        1,              // kernel_size
        1,              // stride
        0,              // padding
        conv_weights,   // weights
        conv_bias       // bias
    };

    return params;
}

void mixed_frequency_attention_forward(
    const std::vector<std::vector<float>>& sal_feat,
    const Tensor3D& edge_feat,
    Tensor3D& output,
    int H, int W, int hidden_size, int num_heads) {

    // Create parameters
    MixedFrequencyAttentionParams params = create_mixed_freq_attention_params(
        hidden_size, num_heads);

    // Call the main implementation
    mixed_frequency_attention_forward(sal_feat, edge_feat, output, params, H, W);
}
