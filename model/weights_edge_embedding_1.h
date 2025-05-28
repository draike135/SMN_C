#pragma once
#include <vector>
#include <string>

extern std::vector<float> edg1_conv1_weights;
extern std::vector<float> edg1_conv1_bias;
extern std::vector<float> edg1_conv2_weights;
extern std::vector<float> edg1_conv2_bias;
extern int hidden_size;

void load_edge_embedding_weights(const std::string &path);
