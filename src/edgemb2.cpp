#include "edgemb2.h"
#include "conv2d_relu.h"
#include "weights_edge_embedding_2.h"
#include <vector>

EdgeEmbedding2Output edge_embedding_2_forward(const Tensor3D &input) {
    Tensor3D x;
    resize_nearest(input, x, 14, 14); // Resize to fixed size

    // Step 1: edge_conv (256 → 128)
    Conv2DParams edge_conv = {256, 128, 3, 1, 1, edge2_conv_weights, std::vector<float>()};
    Tensor3D edge_feat;
    conv2d_forward(x, edge_feat, edge_conv, false); // no ReLU

    // Step 2: edge_mask block (128 → 128)
    Conv2DParams mask_conv1 = {128, 128, 1, 1, 0, edge2_mask1_weights, std::vector<float>()};
    Tensor3D mask1;
    conv2d_forward(edge_feat, mask1, mask_conv1, true); // ReLU

    Conv2DParams mask_conv2 = {128, 128, 1, 1, 0, edge2_mask2_weights, edge2_mask2_bias};
    Tensor3D mask2;
    conv2d_forward(mask1, mask2, mask_conv2, false);
    sigmoid_forward(mask2);

    // Step 3: Enhance (128 → 128)
    Conv2DParams enhance_conv = {128, 128, 3, 1, 1, edge2_enhance_weights, edge2_enhance_bias};
    Tensor3D enhanced;
    conv2d_forward(edge_feat, enhanced, enhance_conv, false);

    Tensor3D combined;
    elementwise_mul_add(edge_feat, mask2, enhanced, combined);
    relu_forward(combined);

    // Step 4: Output conv (128 → 1)
    Conv2DParams out_conv = {128, 1, 3, 1, 1, edge2_out_weights, std::vector<float>()};
    Tensor3D edge_out;
    conv2d_forward(combined, edge_out, out_conv, false);

    return {combined, edge_out};
}
