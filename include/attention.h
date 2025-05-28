#ifndef ATTENTION_H
#define ATTENTION_H

#include "conv2d_relu.h"
#include <cmath>

struct AttentionParams {
    int dim;
    int num_heads;
    int head_dim;
    float scale;
    std::vector<float> qkv_weights;  // Combined Q, K, V weights
    std::vector<float> qkv_bias;
    std::vector<float> proj_weights; // Output projection weights
    std::vector<float> proj_bias;
};

struct MixedFrequencyAttentionParams {
    AttentionParams sa_params;  // Self-attention parameters
    AttentionParams ca_params;  // Cross-attention parameters
    Conv2DParams conv_params;   // 1x1 convolution parameters
};

// Core attention operations
void attention_forward(
    const std::vector<std::vector<float>>& query_input,     // [N, C] - for self-attention, same as key/value
    const std::vector<std::vector<float>>& key_value_input, // [N, C] - for cross-attention
    std::vector<std::vector<float>>& output,                // [N, C]
    const AttentionParams& params,
    bool is_self_attention = true
);

// Squaremax activation (replaces softmax)
void squaremax_forward(
    std::vector<std::vector<float>>& input,  // [N, N] attention scores
    int dim = -1
);

// Mixed frequency attention module
void mixed_frequency_attention_forward(
    const std::vector<std::vector<float>>& sal_feat,  // [H*W, C] spectral saliency features
    const Tensor3D& edge_feat,                        // [C, H, W] edge features
    Tensor3D& output,                                 // [C, H, W] refined features
    const MixedFrequencyAttentionParams& params,
    int H, int W
);

// Helper functions
void reshape_hwc_to_chw(
    const std::vector<std::vector<float>>& input, // [H*W, C]
    Tensor3D& output,                             // [C, H, W]
    int H, int W, int C
);

void reshape_chw_to_hwc(
    const Tensor3D& input,                        // [C, H, W]
    std::vector<std::vector<float>>& output,      // [H*W, C]
    int H, int W, int C
);

#endif // ATTENTION_H
