#ifndef CONV2D_RELU_H
#define CONV2D_RELU_H

#include <vector>

using Tensor3D = std::vector<std::vector<std::vector<float>>>;

struct Conv2DParams {
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;
    std::vector<float> weights; // Flattened weights [out*in*k*k]
    std::vector<float> bias;
};

struct ConvTranspose2DParams {
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;
    std::vector<std::vector<std::vector<std::vector<float>>>> weights; // [in][out][k][k]
    std::vector<float> bias;
};

// Core convolution and activation functions
void conv2d_forward(const Tensor3D &input, Tensor3D &output, const Conv2DParams &params, bool apply_relu);
void relu_forward(Tensor3D &tensor);

// Pooling and resizing functions
void maxpool2d_forward(const Tensor3D &input, Tensor3D &output, int kernel, int stride, int padding);
void resize_nearest(const Tensor3D &input, Tensor3D &output, int targetH, int targetW);

// Additional activation and operations
void sigmoid_forward(Tensor3D &tensor);
void elementwise_mul_add(const Tensor3D &a, const Tensor3D &b, const Tensor3D &add, Tensor3D &out);
void add_tensors(const Tensor3D &a, const Tensor3D &b, Tensor3D &out);

// Decoder-specific functions
void layernorm_forward(Tensor3D &input, float eps = 1e-6f);
void batchnorm2d_forward(
    const Tensor3D &input,
    Tensor3D &output,
    const std::vector<float> &weight,        // gamma (scale parameter)
    const std::vector<float> &bias,          // beta (shift parameter)
    const std::vector<float> &running_mean,  // saved running mean
    const std::vector<float> &running_var,   // saved running variance
    float eps = 1e-5f                        // epsilon for numerical stability
);
void concat_channels(const Tensor3D &a, const Tensor3D &b, Tensor3D &out);
void conv_transpose2d_forward(const Tensor3D &input, Tensor3D &output, const ConvTranspose2DParams &params);

#endif // CONV2D_RELU_H
