#include "weights_specemb.h"
#include "utils.h"

// ResNet weights
std::vector<float> conv1_weights;
std::vector<float> conv1_bias;
std::vector<float> conv2_weights;
std::vector<float> conv2_bias;
std::vector<float> conv3_weights;
std::vector<float> conv3_bias;
std::vector<float> conv4_weights;
std::vector<float> conv4_bias;
std::vector<float> conv5_weights;
std::vector<float> conv5_bias;
std::vector<float> conv6_weights;
std::vector<float> conv6_bias;
std::vector<float> conv7_weights;
std::vector<float> conv7_bias;
std::vector<float> conv8_weights;
std::vector<float> conv8_bias;
std::vector<float> conv9_weights;
std::vector<float> conv9_bias;
std::vector<float> conv10_weights;
std::vector<float> conv10_bias;
std::vector<float> conv11_weights;
std::vector<float> conv11_bias;

// Downsample weights
std::vector<float> down1_weights;
std::vector<float> down1_bias;
std::vector<float> down2_weights;
std::vector<float> down2_bias;

// Patch projection weights
std::vector<float> patch_weights;
std::vector<float> patch_bias;

// Position embeddings
std::vector<std::vector<float>> pos_embedding;

void load_specemb_weights(const std::string &path) {
    // Load ResNet weights
    conv1_weights = load_binary_file(path + "/conv1_weights.bin");
    conv1_bias = load_binary_file(path + "/conv1_bias.bin");
    conv2_weights = load_binary_file(path + "/conv2_weights.bin");
    conv2_bias = load_binary_file(path + "/conv2_bias.bin");
    conv3_weights = load_binary_file(path + "/conv3_weights.bin");
    conv3_bias = load_binary_file(path + "/conv3_bias.bin");
    conv4_weights = load_binary_file(path + "/conv4_weights.bin");
    conv4_bias = load_binary_file(path + "/conv4_bias.bin");
    conv5_weights = load_binary_file(path + "/conv5_weights.bin");
    conv5_bias = load_binary_file(path + "/conv5_bias.bin");
    conv6_weights = load_binary_file(path + "/conv6_weights.bin");
    conv6_bias = load_binary_file(path + "/conv6_bias.bin");
    conv7_weights = load_binary_file(path + "/conv7_weights.bin");
    conv7_bias = load_binary_file(path + "/conv7_bias.bin");
    conv8_weights = load_binary_file(path + "/conv8_weights.bin");
    conv8_bias = load_binary_file(path + "/conv8_bias.bin");
    conv9_weights = load_binary_file(path + "/conv9_weights.bin");
    conv9_bias = load_binary_file(path + "/conv9_bias.bin");
    conv10_weights = load_binary_file(path + "/conv10_weights.bin");
    conv10_bias = load_binary_file(path + "/conv10_bias.bin");
    conv11_weights = load_binary_file(path + "/conv11_weights.bin");
    conv11_bias = load_binary_file(path + "/conv11_bias.bin");

    // Load downsample weights
    down1_weights = load_binary_file(path + "/down1_weights.bin");
    down1_bias = load_binary_file(path + "/down1_bias.bin");
    down2_weights = load_binary_file(path + "/down2_weights.bin");
    down2_bias = load_binary_file(path + "/down2_bias.bin");

    // Load patch projection weights
    patch_weights = load_binary_file(path + "/patch_weights.bin");
    patch_bias = load_binary_file(path + "/patch_bias.bin");

    // Load position embeddings (you'll need to implement this based on your format)
    // For now, initialize with zeros - replace with actual loading code
    pos_embedding.resize(196, std::vector<float>(256, 0.0f)); // 14*14 = 196, 256 channels
}
