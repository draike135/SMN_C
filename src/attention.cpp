#include "attention.h"
#include <algorithm>
#include <cmath>

void squaremax_forward(std::vector<std::vector<float>>& input, int dim) {
    if (dim == -1) dim = input[0].size() - 1; // Last dimension

    for (size_t i = 0; i < input.size(); ++i) {
        // Square positive values (clamp to 0 minimum)
        float sum = 0.0f;
        for (size_t j = 0; j < input[i].size(); ++j) {
            input[i][j] = std::max(0.0f, input[i][j]);
            input[i][j] = input[i][j] * input[i][j]; // Square
            sum += input[i][j];
        }

        // Normalize
        sum += 1e-8f; // eps to avoid division by zero
        for (size_t j = 0; j < input[i].size(); ++j) {
            input[i][j] /= sum;
        }
    }
}

void attention_forward(
    const std::vector<std::vector<float>>& query_input,
    const std::vector<std::vector<float>>& key_value_input,
    std::vector<std::vector<float>>& output,
    const AttentionParams& params,
    bool is_self_attention) {

    int N = query_input.size();    // Sequence length
    int C = params.dim;            // Feature dimension
    int num_heads = params.num_heads;
    int head_dim = params.head_dim;

    // QKV projection
    std::vector<std::vector<float>> Q(N, std::vector<float>(C));
    std::vector<std::vector<float>> K(N, std::vector<float>(C));
    std::vector<std::vector<float>> V(N, std::vector<float>(C));

    // Apply linear transformation for Q (from query_input)
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            Q[n][c] = params.qkv_bias[c]; // Q bias
            for (int in_c = 0; in_c < C; ++in_c) {
                Q[n][c] += query_input[n][in_c] * params.qkv_weights[c * C + in_c];
            }
        }
    }

    // Apply linear transformation for K, V (from key_value_input)
    const auto& kv_input = is_self_attention ? query_input : key_value_input;
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            K[n][c] = params.qkv_bias[C + c]; // K bias
            V[n][c] = params.qkv_bias[2 * C + c]; // V bias
            for (int in_c = 0; in_c < C; ++in_c) {
                K[n][c] += kv_input[n][in_c] * params.qkv_weights[(C + c) * C + in_c];
                V[n][c] += kv_input[n][in_c] * params.qkv_weights[(2 * C + c) * C + in_c];
            }
        }
    }

    // Multi-head attention computation
    output.resize(N, std::vector<float>(C, 0.0f));

    for (int h = 0; h < num_heads; ++h) {
        int start_dim = h * head_dim;
        int end_dim = start_dim + head_dim;

        // Compute attention scores Q @ K^T
        std::vector<std::vector<float>> attn_scores(N, std::vector<float>(N, 0.0f));

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                for (int d = start_dim; d < end_dim; ++d) {
                    attn_scores[i][j] += Q[i][d] * K[j][d] * params.scale;
                }
            }
        }

        // Apply squaremax instead of softmax
        squaremax_forward(attn_scores);

        // Apply attention to values
        for (int i = 0; i < N; ++i) {
            for (int d = start_dim; d < end_dim; ++d) {
                for (int j = 0; j < N; ++j) {
                    output[i][d] += attn_scores[i][j] * V[j][d];
                }
            }
        }
    }

    // Output projection
    std::vector<std::vector<float>> projected(N, std::vector<float>(C));
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            projected[n][c] = params.proj_bias[c];
            for (int in_c = 0; in_c < C; ++in_c) {
                projected[n][c] += output[n][in_c] * params.proj_weights[c * C + in_c];
            }
        }
    }

    output = projected;
}

void reshape_chw_to_hwc(
    const Tensor3D& input,
    std::vector<std::vector<float>>& output,
    int H, int W, int C) {

    output.resize(H * W, std::vector<float>(C));

    for (int c = 0; c < C; ++c) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                int index = h * W + w;
                output[index][c] = input[c][h][w];
            }
        }
    }
}

void reshape_hwc_to_chw(
    const std::vector<std::vector<float>>& input,
    Tensor3D& output,
    int H, int W, int C) {

    output.resize(C, std::vector<std::vector<float>>(H, std::vector<float>(W)));

    for (int c = 0; c < C; ++c) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                int index = h * W + w;
                output[c][h][w] = input[index][c];
            }
        }
    }
}

void mixed_frequency_attention_forward(
    const std::vector<std::vector<float>>& sal_feat,
    const Tensor3D& edge_feat,
    Tensor3D& output,
    const MixedFrequencyAttentionParams& params,
    int H, int W) {

    int C = sal_feat[0].size();
    int half_C = C / 2;

    // Extract first half of sal_feat for self-attention
    std::vector<std::vector<float>> sal_feat_first_half(sal_feat.size(),
                                                        std::vector<float>(half_C));
    for (size_t n = 0; n < sal_feat.size(); ++n) {
        for (int c = 0; c < half_C; ++c) {
            sal_feat_first_half[n][c] = sal_feat[n][c];
        }
    }

    // Extract second half of sal_feat for cross-attention
    std::vector<std::vector<float>> sal_feat_second_half(sal_feat.size(),
                                                         std::vector<float>(half_C));
    for (size_t n = 0; n < sal_feat.size(); ++n) {
        for (int c = 0; c < half_C; ++c) {
            sal_feat_second_half[n][c] = sal_feat[n][half_C + c];
        }
    }

    // Convert edge_feat from [C, H, W] to [H*W, C] format
    std::vector<std::vector<float>> edge_feat_hwc;
    reshape_chw_to_hwc(edge_feat, edge_feat_hwc, H, W, edge_feat.size());

    // Apply self-attention (SA)
    std::vector<std::vector<float>> attn_sa_result;
    attention_forward(sal_feat_first_half, sal_feat_first_half,
                     attn_sa_result, params.sa_params, true);

    // Apply cross-attention (CA)
    std::vector<std::vector<float>> attn_ca_result;
    attention_forward(edge_feat_hwc, sal_feat_second_half,
                     attn_ca_result, params.ca_params, false);

    // Convert back to [C, H, W] format
    Tensor3D attn_sa_chw, attn_ca_chw;
    reshape_hwc_to_chw(attn_sa_result, attn_sa_chw, H, W, half_C);
    reshape_hwc_to_chw(attn_ca_result, attn_ca_chw, H, W, half_C);

    // Concatenate features along channel dimension
    Tensor3D concatenated(C, std::vector<std::vector<float>>(H, std::vector<float>(W)));

    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            // First half from SA
            for (int c = 0; c < half_C; ++c) {
                concatenated[c][h][w] = attn_sa_chw[c][h][w];
            }
            // Second half from CA
            for (int c = 0; c < half_C; ++c) {
                concatenated[half_C + c][h][w] = attn_ca_chw[c][h][w];
            }
        }
    }

    // Apply 1x1 convolution + ReLU
    conv2d_forward(concatenated, output, params.conv_params, true); // Apply ReLU
}
