#include "conv2d_relu.h"
#include <cmath>
#include <algorithm>

void conv2d_forward(const Tensor3D &input,
                    Tensor3D &output,
                    const Conv2DParams &params,
                    bool apply_relu) {
    int in_c = params.in_channels;
    int out_c = params.out_channels;
    int k = params.kernel_size;
    int stride = params.stride;
    int pad = params.padding;

    int in_h = input[0].size();
    int in_w = input[0][0].size();
    int out_h = (in_h + 2 * pad - k) / stride + 1;
    int out_w = (in_w + 2 * pad - k) / stride + 1;

    output.resize(out_c, std::vector<std::vector<float>>(out_h, std::vector<float>(out_w, 0.0f)));

    // Check if bias exists
    bool has_bias = !params.bias.empty();

    for (int oc = 0; oc < out_c; ++oc) {
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                // Initialize with bias if available, otherwise start with 0
                float sum = has_bias ? params.bias[oc] : 0.0f;

                for (int ic = 0; ic < in_c; ++ic) {
                    for (int kh = 0; kh < k; ++kh) {
                        for (int kw = 0; kw < k; ++kw) {
                            int ih = oh * stride + kh - pad;
                            int iw = ow * stride + kw - pad;
                            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                int weight_idx = ((oc * in_c + ic) * k + kh) * k + kw;
                                float weight = params.weights[weight_idx];
                                sum += weight * input[ic][ih][iw];
                            }
                        }
                    }
                }
                output[oc][oh][ow] = apply_relu ? std::max(0.0f, sum) : sum;
            }
        }
    }
}

void relu_forward(Tensor3D &tensor) {
    for (auto &channel : tensor) {
        for (auto &row : channel) {
            for (auto &val : row) {
                val = std::max(0.0f, val);
            }
        }
    }
}

void maxpool2d_forward(
    const Tensor3D &input,
    Tensor3D &output,
    int kernel, int stride, int padding)
{
    int C = input.size();
    int H = input[0].size();
    int W = input[0][0].size();

    int H_out = (H + 2 * padding - kernel) / stride + 1;
    int W_out = (W + 2 * padding - kernel) / stride + 1;

    output.resize(C, std::vector<std::vector<float>>(H_out, std::vector<float>(W_out, 0)));

    for (int c = 0; c < C; ++c) {
        for (int i = 0; i < H_out; ++i) {
            for (int j = 0; j < W_out; ++j) {
                float max_val = -1e30f;
                for (int ki = 0; ki < kernel; ++ki) {
                    for (int kj = 0; kj < kernel; ++kj) {
                        int y = i * stride + ki - padding;
                        int x = j * stride + kj - padding;
                        if (y >= 0 && y < H && x >= 0 && x < W) {
                            max_val = std::max(max_val, input[c][y][x]);
                        }
                    }
                }
                output[c][i][j] = max_val;
            }
        }
    }
}

void resize_nearest(
    const Tensor3D &input,
    Tensor3D &output,
    int targetH,
    int targetW)
{
    int C = input.size();
    int H = input[0].size();
    int W = input[0][0].size();

    output.resize(C, std::vector<std::vector<float>>(targetH, std::vector<float>(targetW)));

    for (int c = 0; c < C; ++c) {
        for (int h2 = 0; h2 < targetH; ++h2) {
            int h1 = std::min(int(h2 * H / targetH), H - 1);
            for (int w2 = 0; w2 < targetW; ++w2) {
                int w1 = std::min(int(w2 * W / targetW), W - 1);
                output[c][h2][w2] = input[c][h1][w1];
            }
        }
    }
}

void sigmoid_forward(Tensor3D &tensor) {
    for (size_t c = 0; c < tensor.size(); ++c) {
        for (size_t h = 0; h < tensor[c].size(); ++h) {
            for (size_t w = 0; w < tensor[c][h].size(); ++w) {
                float val = tensor[c][h][w];
                tensor[c][h][w] = 1.0f / (1.0f + std::exp(-val));
            }
        }
    }
}

void elementwise_mul_add(
    const Tensor3D &a,
    const Tensor3D &b,
    const Tensor3D &add,
    Tensor3D &out)
{
    int C = a.size();
    int H = a[0].size();
    int W = a[0][0].size();

    out.resize(C, std::vector<std::vector<float>>(H, std::vector<float>(W)));

    for (int c = 0; c < C; ++c)
        for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w)
                out[c][h][w] = a[c][h][w] * b[c][h][w] + add[c][h][w];
}

void add_tensors(const Tensor3D &a, const Tensor3D &b, Tensor3D &out) {
    int C = a.size();
    int H = a[0].size();
    int W = a[0][0].size();
    out.resize(C, std::vector<std::vector<float>>(H, std::vector<float>(W)));

    for (int c = 0; c < C; ++c)
        for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w)
                out[c][h][w] = a[c][h][w] + b[c][h][w];
}

void batchnorm2d_forward(
    const Tensor3D &input,
    Tensor3D &output,
    const std::vector<float> &weight,    // gamma (scale parameter)
    const std::vector<float> &bias,      // beta (shift parameter)
    const std::vector<float> &running_mean,
    const std::vector<float> &running_var,
    float eps = 1e-5f)
{
    int C = input.size();
    int H = input[0].size();
    int W = input[0][0].size();

    output.resize(C, std::vector<std::vector<float>>(H, std::vector<float>(W)));

    // Apply batch norm per channel: output = weight * (input - mean) / sqrt(var + eps) + bias
    for (int c = 0; c < C; ++c) {
        float mean = running_mean[c];
        float var = running_var[c];
        float scale = weight[c] / std::sqrt(var + eps);  // weight / sqrt(var + eps)
        float shift = bias[c] - mean * scale;            // bias - mean * scale

        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                output[c][h][w] = input[c][h][w] * scale + shift;
            }
        }
    }
}
