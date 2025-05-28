#include "utils.h"
#include <fstream>
#include <stdexcept>

std::vector<float> load_binary_file(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open file: " + filename);

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0);

    std::vector<float> data(size / sizeof(float));
    file.read(reinterpret_cast<char*>(data.data()), size);

    return data;
}

std::vector<std::vector<std::vector<float>>> load_image_bin(const std::string &filename, int C, int H, int W) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Failed to open image file: " + filename);

    std::vector<std::vector<std::vector<float>>> image(C, std::vector<std::vector<float>>(H, std::vector<float>(W)));

    for (int c = 0; c < C; ++c)
        for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w) {
                float val;
                file.read(reinterpret_cast<char*>(&val), sizeof(float));
                image[c][h][w] = val;
            }

    return image;
}
