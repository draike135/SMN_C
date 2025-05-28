#include "weights_edge_embedding_2.h"
#include "utils.h"

std::vector<float> edge2_conv_weights;
//std::vector<float> edge2_conv_bias;
std::vector<float> edge2_mask1_weights;
//std::vector<float> edge2_mask1_bias;
std::vector<float> edge2_mask2_weights;
std::vector<float> edge2_mask2_bias;
std::vector<float> edge2_enhance_weights;
std::vector<float> edge2_enhance_bias;
std::vector<float> edge2_out_weights;
//std::vector<float> edge2_out_bias;

void load_edge_embedding_2_weights(const std::string &path) {
    edge2_conv_weights = load_binary_file(path + "/edge2_conv_weights.bin");
    //edge2_conv_bias = load_binary_file(path + "/edge2_conv_bias.bin");
    edge2_mask1_weights = load_binary_file(path + "/edge2_mask1_weights.bin");
    //edge2_mask1_bias = load_binary_file(path + "/edge2_mask1_bias.bin");
    edge2_mask2_weights = load_binary_file(path + "/edge2_mask2_weights.bin");
    edge2_mask2_bias = load_binary_file(path + "/edge2_mask2_bias.bin");
    edge2_enhance_weights = load_binary_file(path + "/edge2_enhance_weights.bin");
    edge2_enhance_bias = load_binary_file(path + "/edge2_enhance_bias.bin");
    edge2_out_weights = load_binary_file(path + "/edge2_out_weights.bin");
    //edge2_out_bias = load_binary_file(path + "/edge2_out_bias.bin");
}
