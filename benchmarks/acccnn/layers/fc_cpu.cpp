#include "../cnn_pipeline.h"
#include <vector>
#include <cmath>
#include <iostream>

//
// ------------------------------------------------------------
// Fully Connected Layer (CPU)
// ------------------------------------------------------------
//
// input:  Tensor(C, H, W) flattened internally
// W:      weights, size = out_dim * in_dim
// B:      biases, size = out_dim
//
// returns: logits[size=out_dim]
// ------------------------------------------------------------

std::vector<float> fc_cpu(const Tensor& input,
                          const std::vector<float>& W,
                          const std::vector<float>& B)
{
    int in_dim  = input.C * input.H * input.W;
    int out_dim = B.size();

    std::vector<float> out(out_dim, 0.0f);

    const float* x = input.data.data();

    for (int o = 0; o < out_dim; ++o) {
        float sum = B[o];
        const float* w_row = &W[o * in_dim];

        for (int i = 0; i < in_dim; ++i) {
            sum += x[i] * w_row[i];
        }
        out[o] = sum;
    }

    return out;
}
