#ifndef WEIGHTS_EDGE_EMBEDDING_2_H
#define WEIGHTS_EDGE_EMBEDDING_2_H

#include <vector>
#include <string>

extern std::vector<float> edge2_conv_weights;
//extern std::vector<float> edge2_conv_bias;
extern std::vector<float> edge2_mask1_weights;
//extern std::vector<float> edge2_mask1_bias;
extern std::vector<float> edge2_mask2_weights;
extern std::vector<float> edge2_mask2_bias;
extern std::vector<float> edge2_enhance_weights;
extern std::vector<float> edge2_enhance_bias;
extern std::vector<float> edge2_out_weights;
//extern std::vector<float> edge2_out_bias;

void load_edge_embedding_2_weights(const std::string &path);

#endif // WEIGHTS_EDGE_EMBEDDING_2_H
