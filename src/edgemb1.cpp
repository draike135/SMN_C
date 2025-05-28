#include "edgemb1.h"
#include "conv2d_relu.h"
#include "weights_edge_embedding_1.h"

Tensor3D edge_embedding_1_forward(const Tensor3D &input) {
    Conv2DParams conv1 = {
        3, 64, 3, 2, 1,
        edg1_conv1_weights,
        edg1_conv1_bias
    };
    Tensor3D out1;
    conv2d_forward(input, out1, conv1, true);

    Conv2DParams conv2 = {
        64, hidden_size, 3, 2, 1,
        edg1_conv2_weights,
        edg1_conv2_bias
    };
    Tensor3D out2;
    conv2d_forward(out1, out2, conv2, true);

    return out2;
}
