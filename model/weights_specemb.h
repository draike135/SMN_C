#ifndef WEIGHTS_SPECEMB_H
#define WEIGHTS_SPECEMB_H

#include <vector>
#include <string>

// ResNet weights
extern std::vector<float> conv1_weights;
extern std::vector<float> conv1_bias;
extern std::vector<float> conv2_weights;
extern std::vector<float> conv2_bias;
extern std::vector<float> conv3_weights;
extern std::vector<float> conv3_bias;
extern std::vector<float> conv4_weights;
extern std::vector<float> conv4_bias;
extern std::vector<float> conv5_weights;
extern std::vector<float> conv5_bias;
extern std::vector<float> conv6_weights;
extern std::vector<float> conv6_bias;
extern std::vector<float> conv7_weights;
extern std::vector<float> conv7_bias;
extern std::vector<float> conv8_weights;
extern std::vector<float> conv8_bias;
extern std::vector<float> conv9_weights;
extern std::vector<float> conv9_bias;
extern std::vector<float> conv10_weights;
extern std::vector<float> conv10_bias;
extern std::vector<float> conv11_weights;
extern std::vector<float> conv11_bias;

// Downsample weights
extern std::vector<float> down1_weights;
extern std::vector<float> down1_bias;
extern std::vector<float> down2_weights;
extern std::vector<float> down2_bias;

// Patch projection weights
extern std::vector<float> patch_weights;
extern std::vector<float> patch_bias;

// Position embeddings
extern std::vector<std::vector<float>> pos_embedding;

void load_specemb_weights(const std::string &path);

#endif // WEIGHTS_SPECEMB_H
