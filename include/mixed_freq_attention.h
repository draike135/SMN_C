#ifndef MIXED_FREQ_ATTENTION_H
#define MIXED_FREQ_ATTENTION_H

#include "attention.h"
#include "conv2d_relu.h"

// High-level interface for mixed frequency attention
void mixed_frequency_attention_forward(
    const std::vector<std::vector<float>>& sal_feat,  // [H*W, C] spectral saliency features
    const Tensor3D& edge_feat,                        // [C, H, W] edge features
    Tensor3D& output,                                 // [C, H, W] refined features
    int H, int W, int hidden_size, int num_heads
);

// Initialize attention parameters
MixedFrequencyAttentionParams create_mixed_freq_attention_params(
    int hidden_size,
    int num_heads
);

#endif // MIXED_FREQ_ATTENTION_H
