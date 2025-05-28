#ifndef WEIGHTS_DECODER_H
#define WEIGHTS_DECODER_H

#include <vector>
#include <string>

// Decoder convolution weights
extern std::vector<float> decoder_conv0_weights;
extern std::vector<float> decoder_conv0_bias;
extern std::vector<float> decoder_conv1_weights;
extern std::vector<float> decoder_conv1_bias;
extern std::vector<float> decoder_conv2_weights;
extern std::vector<float> decoder_conv2_bias;
extern std::vector<float> decoder_conv3_weights;
extern std::vector<float> decoder_conv3_bias;
extern std::vector<float> decoder_conv4_weights;
extern std::vector<float> decoder_conv4_bias;

// Transposed convolution weights
extern std::vector<std::vector<std::vector<std::vector<float>>>> decoder_upconv1_weights;
//extern std::vector<float> decoder_upconv1_bias;
extern std::vector<std::vector<std::vector<std::vector<float>>>> decoder_upconv2_weights;
//extern std::vector<float> decoder_upconv2_bias;

// Configuration
extern int decoder_embed_dim;
extern int decoder_hidden_size;

void load_decoder_weights(const std::string &path);

#endif // WEIGHTS_DECODER_H
