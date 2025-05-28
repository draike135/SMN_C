#include "weights_edge_embedding_1.h"
#include "utils.h"  // <- include the loader function

std::vector<float> edg1_conv1_weights;
std::vector<float> edg1_conv1_bias;
std::vector<float> edg1_conv2_weights;
std::vector<float> edg1_conv2_bias;
int hidden_size = 256;

void load_edge_embedding_weights(const std::string &path) {
    edg1_conv1_weights = load_binary_file(path + "/conv1_weights.bin");
    edg1_conv1_bias = load_binary_file(path + "/conv1_bias.bin");
    edg1_conv2_weights = load_binary_file(path + "/conv2_weights.bin");
    edg1_conv2_bias = load_binary_file(path + "/conv2_bias.bin");

    hidden_size = 256;  // fallback
}
