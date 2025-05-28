#ifndef SPECEMB_H
#define SPECEMB_H

#include "conv2d_relu.h"
#include <vector>

struct ResNetOutput {
    Tensor3D final_feature; // Last feature map (256 x H x W)
    Tensor3D feature1;      // After layer1
    Tensor3D feature2;      // After layer2
    Tensor3D feature3;      // After layer3
};

struct SpecEmbeddingOutput {
    std::vector<std::vector<float>> embedding; // [H*W][C]
    Tensor3D feature1;                         // [C][H][W]
    Tensor3D feature2;
    Tensor3D feature3;
};

ResNetOutput resnet_forward(const Tensor3D &input);
SpecEmbeddingOutput spec_embedding_forward(const Tensor3D &input);

#endif // SPECEMB_H
