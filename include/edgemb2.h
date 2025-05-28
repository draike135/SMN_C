#ifndef EDGEMB2_H
#define EDGEMB2_H

#include "conv2d_relu.h"

struct EdgeEmbedding2Output {
    Tensor3D edge_feature; // [256][H][W]
    Tensor3D edge_output;  // [1][H][W]
};

EdgeEmbedding2Output edge_embedding_2_forward(const Tensor3D &input);

#endif // EDGEMB2_H
