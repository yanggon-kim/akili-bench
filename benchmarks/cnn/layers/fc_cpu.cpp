#include "../cnn_pipeline.h"
#include "../common.h"
#include <vector>
#include <cmath>

// FC weights layout (what your export script already writes):
//   - Keras Dense:  W_keras shape = (in_dim, out_dim)
//   - export_fc:    Wfile[o, i] = W_keras[i, o]
//   - flattened as: Wfile[o*in_dim + i]  (row-major [out][in])
//
// Here we must:
//   1) Flatten activations in **(y, x, c)** order to match Keras Flatten
//   2) Use the same in_dim index i when reading Wfile[o*in_dim + i]

std::vector<float> fc_cpu(const Tensor& input,
                          const std::vector<float>& W,
                          const std::vector<float>& b)
{
    int C = input.C;
    int H = input.H;
    int Ww = input.W;

    int in_dim  = C * H * Ww;          // 8 * 13 * 13 = 1352
    int out_dim = static_cast<int>(b.size()); // 10

    std::vector<float> logits(out_dim, 0.0f);

    for (int o = 0; o < out_dim; ++o) {
        float sum = b[o];
        int idx = 0; // feature index i in [0, in_dim)

        // IMPORTANT: iterate in (y, x, c) order to match Keras Flatten on NHWC
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < Ww; ++x) {
                for (int c = 0; c < C; ++c) {
                    // Tensor storage is CHW: [c][y][x]
                    float v = input.data[c * H * Ww + y * Ww + x];

                    // Weight W[o, idx]
                    float w = W[o * in_dim + idx];

                    sum += v * w;
                    ++idx;
                }
            }
        }

        logits[o] = sum;
    }

    return logits;
}
