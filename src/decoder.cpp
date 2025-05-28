#include "decoder.h"
#include "conv2d_relu.h"
#include "weights_decoder.h"

DecoderParams create_decoder_params() {
    DecoderParams params;

    params.hidden_size = decoder_hidden_size;

    // Conv0: embed_dim -> hidden_size
    params.conv0 = {decoder_embed_dim, decoder_hidden_size, 3, 1, 1,
                    decoder_conv0_weights, decoder_conv0_bias};

    // Conv1: hidden_size -> hidden_size
    params.conv1 = {decoder_hidden_size, decoder_hidden_size, 3, 1, 1,
                    decoder_conv1_weights, decoder_conv1_bias};

    // Conv2: hidden_size -> hidden_size
    params.conv2 = {decoder_hidden_size, decoder_hidden_size, 3, 1, 1,
                    decoder_conv2_weights, decoder_conv2_bias};

    // Conv3: hidden_size -> hidden_size
    params.conv3 = {decoder_hidden_size, decoder_hidden_size, 3, 1, 1,
                    decoder_conv3_weights, decoder_conv3_bias};

    // Conv4: hidden_size -> 1 (output)
    params.conv4 = {decoder_hidden_size, 1, 3, 1, 1,
                    decoder_conv4_weights, std::vector<float>()};

    // Transposed convolutions (parameters depend on backbone)
    // For resnet18/pvt_v2_b1:
    int tc = (decoder_hidden_size == 256) ? 128 : decoder_hidden_size;

    params.upconv1 = {128, tc, 4, 4, 0, decoder_upconv1_weights, decoder_upconv1_bias};
    params.upconv2 = {64, tc, 4, 2, 1, decoder_upconv2_weights, decoder_upconv2_bias};

    return params;
}

Tensor3D decoder_forward(
    const Tensor3D &input,
    const Tensor3D &edge_feat,
    const Tensor3D &spec_feat_0,
    const Tensor3D &spec_feat_1,
    const DecoderParams &params)
{
    Tensor3D x;

    // === LayerNorm ===
    x = input;
    layernorm_forward(x);

    // === conv0 + ReLU ===
    conv2d_forward(x, x, params.conv0, true);

    // === Upsample ×2 ===
    resize_nearest(x, x, x[0].size() * 2, x[0][0].size() * 2);

    // === conv1 + ReLU ===
    conv2d_forward(x, x, params.conv1, true);

    // === Upsample ×2 ===
    resize_nearest(x, x, x[0].size() * 2, x[0][0].size() * 2);

    // === Add edge features ===
    Tensor3D combined;
    add_tensors(x, edge_feat, combined);

    // === conv2 + ReLU ===
    conv2d_forward(combined, x, params.conv2, true);

    // === Upsample ×2 ===
    resize_nearest(x, x, x[0].size() * 2, x[0][0].size() * 2);

    // === Transposed convolutions ===
    Tensor3D up1, up2;
    conv_transpose2d_forward(spec_feat_1, up1, params.upconv1);
    conv_transpose2d_forward(spec_feat_0, up2, params.upconv2);

    // === Combine spec features ===
    Tensor3D spec_combined;
    if (params.hidden_size == 256) {  // Fixed comparison - was params.hidden_size / 2 == 128
        concat_channels(up1, up2, spec_combined);
        add_tensors(x, spec_combined, combined);
    } else {
        add_tensors(x, up1, combined);
        add_tensors(combined, up2, combined);
    }

    // === conv3 + ReLU ===
    conv2d_forward(combined, x, params.conv3, true);

    // === Final upsample ===
    resize_nearest(x, x, x[0].size() * 2, x[0][0].size() * 2);

    // === Output conv (no ReLU) ===
    Tensor3D out;
    conv2d_forward(x, out, params.conv4, false);

    return out;
}
