#include "specemb.h"
#include "conv2d_relu.h"
#include "weights_specemb.h"
#include <vector>

ResNetOutput resnet_forward(const Tensor3D &input) {
    Tensor3D x, identity, out;

    // Initial Conv (3 -> 64)
    Conv2DParams conv1 = {3, 64, 7, 2, 3, conv1_weights, conv1_bias};
    conv2d_forward(input, x, conv1, true);

    // MaxPool
    maxpool2d_forward(x, x, 3, 2, 1);

    // === Layer 1, Block 1 ===
    identity = x;
    Conv2DParams conv2 = {64, 64, 3, 1, 1, conv2_weights, conv2_bias};
    conv2d_forward(x, out, conv2, true);
    Conv2DParams conv3 = {64, 64, 3, 1, 1, conv3_weights, conv3_bias};
    conv2d_forward(out, out, conv3, false);
    add_tensors(out, identity, x);
    relu_forward(x);

    // === Layer 1, Block 2 ===
    identity = x;
    Conv2DParams conv4 = {64, 64, 3, 1, 1, conv4_weights, conv4_bias};
    conv2d_forward(x, out, conv4, true);
    Conv2DParams conv5 = {64, 64, 3, 1, 1, conv5_weights, conv5_bias};
    conv2d_forward(out, out, conv5, false);
    add_tensors(out, identity, x);
    relu_forward(x);

    Tensor3D feat1 = x;

    // === Layer 2, Block 1 (downsample) ===
    identity.clear();
    Conv2DParams down1 = {64, 128, 1, 2, 0, down1_weights, down1_bias};
    conv2d_forward(x, identity, down1, false);

    Conv2DParams conv6 = {64, 128, 3, 2, 1, conv6_weights, conv6_bias};
    conv2d_forward(x, out, conv6, true);
    Conv2DParams conv7 = {128, 128, 3, 1, 1, conv7_weights, conv7_bias};
    conv2d_forward(out, out, conv7, false);
    add_tensors(out, identity, x);
    relu_forward(x);

    // === Layer 2, Block 2 ===
    identity = x;
    Conv2DParams conv8 = {128, 128, 3, 1, 1, conv8_weights, conv8_bias};
    conv2d_forward(x, out, conv8, true);
    Conv2DParams conv9 = {128, 128, 3, 1, 1, conv9_weights, conv9_bias};
    conv2d_forward(out, out, conv9, false);
    add_tensors(out, identity, x);
    relu_forward(x);

    Tensor3D feat2 = x;

    // === Layer 3, Block 1 (downsample) ===
    identity.clear();
    Conv2DParams down2 = {128, 256, 1, 2, 0, down2_weights, down2_bias};
    conv2d_forward(x, identity, down2, false);

    Conv2DParams conv10 = {128, 256, 3, 2, 1, conv10_weights, conv10_bias};
    conv2d_forward(x, out, conv10, true);
    Conv2DParams conv11 = {256, 256, 3, 1, 1, conv11_weights, conv11_bias};
    conv2d_forward(out, out, conv11, false);
    add_tensors(out, identity, x);
    relu_forward(x);

    Tensor3D feat3 = x;

    return {x, feat1, feat2, feat3};
}

SpecEmbeddingOutput spec_embedding_forward(const Tensor3D &input) {
    // Run the ResNet backbone
    ResNetOutput resnet_out = resnet_forward(input);
    auto x = resnet_out.final_feature;

    // Patch projection (1x1 Conv2D)
    Conv2DParams patch_proj = {
        256, 256, 1, 1, 0,
        patch_weights,
        patch_bias
    };
    Tensor3D x_proj;
    conv2d_forward(x, x_proj, patch_proj, false); // No ReLU

    // Flatten spatial dimensions: [C][H][W] → [H*W][C]
    int C = x_proj.size();
    int H = x_proj[0].size();
    int W = x_proj[0][0].size();
    std::vector<std::vector<float>> flattened(H * W, std::vector<float>(C));

    for (int c = 0; c < C; ++c) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                int index = h * W + w;
                flattened[index][c] = x_proj[c][h][w];
            }
        }
    }

    // Add position embeddings: flattened += pos_embedding
    for (int i = 0; i < H * W; ++i) {
        for (int j = 0; j < C; ++j) {
            flattened[i][j] += pos_embedding[i][j];
        }
    }

    return {flattened, resnet_out.feature1, resnet_out.feature2, resnet_out.feature3};
}
