#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>

std::vector<float> load_binary_file(const std::string &filename);
std::vector<std::vector<std::vector<float>>> load_image_bin(const std::string &filename, int C, int H, int W);


#endif
