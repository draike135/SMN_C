#include "weights_decoder.h"
#include "utils.h"

// Decoder convolution weights
std::vector<float> decoder_conv0_weights;
std::vector<float> decoder_conv0_bias;
std::vector<float> decoder_conv1_weights;
std::vector<float> decoder_conv1_bias;
std::vector<float> decoder_conv2_weights;
std::vector<float> decoder_conv2_bias;
std::vector<float> decoder_conv3_weights;
std::vector<float> decoder_conv3_bias;
std::vector<float> decoder_conv4_weights;
//std::vector<float> decoder_conv4_bias;

// Transposed convolution weights
std::vector<std::vector<std::vector<std::vector<float>>>> decoder_upconv1_weights;
//std::vector<float> decoder_upconv1_bias;
std::vector<std::vector<std::vector<std::vector<float>>>> decoder_upconv2_weights;
//std::vector<float> decoder_upconv2_bias;

// Configuration
int decoder_embed_dim = 1024;
int decoder_hidden_size = 256;

void load_decoder_weights(const std::string &path) {
    // Load regular convolution weights
    decoder_conv0_weights = load_binary_file(path + "/conv0_weights.bin");
    decoder_conv0_bias = load_binary_file(path + "/conv0_bias.bin");
    decoder_conv1_weights = load_binary_file(path + "/conv1_weights.bin");
    decoder_conv1_bias = load_binary_file(path + "/conv1_bias.bin");
    decoder_conv2_weights = load_binary_file(path + "/conv2_weights.bin");
    decoder_conv2_bias = load_binary_file(path + "/conv2_bias.bin");
    decoder_conv3_weights = load_binary_file(path + "/conv3_weights.bin");
    decoder_conv3_bias = load_binary_file(path + "/conv3_bias.bin");
    decoder_conv4_weights = load_binary_file(path + "/conv4_weights.bin");
    //decoder_conv4_bias = load_binary_file(path + "/conv4_bias.bin");

    // Load transposed convolution biases
    //decoder_upconv1_bias = load_binary_file(path + "/upconv1_bias.bin");
    //ecoder_upconv2_bias = load_binary_file(path + "/upconv2_bias.bin");

    // Note: For transposed convolution weights, you'll need to implement
    // a function to load 4D weights from binary files
    // For now, initialize with empty vectors - replace with actual loading

    // Example structure for upconv1: [128][tc][4][4] where tc depends on backbone
    // Example structure for upconv2: [64][tc][4][4] or [64][tc][2][2] depending on stride

    // You'll need to implement load_4d_weights function based on your weight file format
    // decoder_upconv1_weights = load_4d_weights(path + "/upconv1_weights.bin", 128, tc, 4, 4);
    // decoder_upconv2_weights = load_4d_weights(path + "/upconv2_weights.bin", 64, tc, kernel, kernel);
}
