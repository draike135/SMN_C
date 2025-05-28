#ifndef DECODER_H
#define DECODER_H

#include "conv2d_relu.h"
#include <vector>
#include <cmath>


struct DecoderParams {
    Conv2DParams conv0;
    Conv2DParams conv1;
    Conv2DParams conv2;
    Conv2DParams conv3;
    Conv2DParams conv4;
    ConvTranspose2DParams upconv1;
    ConvTranspose2DParams upconv2;
    int hidden_size;
};

// Function declarations
DecoderParams create_decoder_params();

Tensor3D decoder_forward(
    const Tensor3D &input,
    const Tensor3D &edge_feat,
    const Tensor3D &spec_feat_0,
    const Tensor3D &spec_feat_1,
    const DecoderParams &params);

#endif // DECODER_H
