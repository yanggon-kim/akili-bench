#include "cnn_pipeline.h"
#include <fstream>
#include <cmath>
#include <iostream>
#include <cstring>

//
// -------------------------------------------------------------
//  Utility: VortexDevice
// -------------------------------------------------------------
//

VortexDevice::VortexDevice() {
    if (vx_dev_open(&dev)) {
        std::cerr << "Failed to open Vortex device\n";
        exit(1);
    }
}

VortexDevice::~VortexDevice() {
    if (dev)
        vx_dev_close(dev);
}

void VortexDevice::check(int err, const char* msg) {
    if (err) {
        std::cerr << "Vortex error: " << msg << ", code=" << err << "\n";
        exit(1);
    }
}

//
// -------------------------------------------------------------
//  Utility: Device memory helpers
// -------------------------------------------------------------
//

uint64_t upload_tensor_to_device(VortexDevice& vx,
                                 const Tensor& t,
                                 vx_buffer_h* buf_out)
{
    size_t nbytes = t.data.size() * sizeof(float);
    vx_buffer_h buf;

    vx.check(vx_mem_alloc(vx.dev, nbytes, VX_MEM_READ, &buf),
             "alloc tensor");
    uint64_t addr;
    vx.check(vx_mem_address(buf, &addr), "tensor address");
    vx.check(vx_copy_to_dev(buf, t.data.data(), 0, nbytes),
             "copy tensor to device");

    *buf_out = buf;
    return addr;
}

uint64_t upload_weights_to_device(VortexDevice& vx,
                                  const std::vector<float>& W,
                                  vx_buffer_h* buf_out)
{
    size_t nbytes = W.size() * sizeof(float);
    vx_buffer_h buf;

    vx.check(vx_mem_alloc(vx.dev, nbytes, VX_MEM_READ, &buf),
             "alloc weights");
    uint64_t addr;
    vx.check(vx_mem_address(buf, &addr), "weights address");
    vx.check(vx_copy_to_dev(buf, W.data(), 0, nbytes),
             "copy weights to dev");

    *buf_out = buf;
    return addr;
}

void download_tensor_from_device(VortexDevice& vx,
                                 Tensor& out,
                                 vx_buffer_h buf)
{
    size_t nbytes = out.data.size() * sizeof(float);
    vx.check(vx_copy_from_dev(out.data.data(), buf, 0, nbytes),
             "download tensor");
}

//
// -------------------------------------------------------------
//  GPU Ops: conv, pool, relu
//  (Implementation arrives from conv_gpu.cpp, pool_gpu.cpp, relu_gpu.cpp)
// -------------------------------------------------------------
//

// NOTE: These functions are defined in separate files.
// The declarations are already in cnn_pipeline.h.

//
// -------------------------------------------------------------
//  CPU Ops: FC and Softmax
// -------------------------------------------------------------
//

// std::vector<float> fc_cpu(const Tensor& input,
//                           const std::vector<float>& W,
//                           const std::vector<float>& B)
// {
//     // input is CHW flattened into 1D
//     int in_dim = input.C * input.H * input.W;
//     int out_dim = B.size(); // e.g., 10

//     std::vector<float> out(out_dim, 0.0f);

//     for (int o = 0; o < out_dim; ++o) {
//         float sum = B[o];
//         for (int i = 0; i < in_dim; ++i) {
//             sum += input.data[i] * W[o * in_dim + i];
//         }
//         out[o] = sum;
//     }
//     return out;
// }

std::vector<float> softmax(const std::vector<float>& logits) {
    std::vector<float> out(logits.size());
    float maxv = -1e30;

    for (float v : logits)
        if (v > maxv) maxv = v;

    float sum = 0.0f;
    for (float v : logits)
        sum += std::exp(v - maxv);

    for (size_t i = 0; i < logits.size(); ++i)
        out[i] = std::exp(logits[i] - maxv) / sum;

    return out;
}

int argmax(const std::vector<float>& v) {
    int idx = 0;
    float best = v[0];
    for (int i = 1; i < (int)v.size(); ++i) {
        if (v[i] > best) {
            best = v[i];
            idx = i;
        }
    }
    return idx;
}

//
// -------------------------------------------------------------
//  .bin weight loading helpers
// -------------------------------------------------------------
//

std::vector<float> load_bin_vector(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "Cannot open: " << path << "\n";
        exit(1);
    }
    f.seekg(0, std::ios::end);
    size_t size = f.tellg();
    f.seekg(0);

    std::vector<float> v(size / sizeof(float));
    f.read(reinterpret_cast<char*>(v.data()), size);
    return v;
}

Tensor load_bin_tensor(const std::string& path, int C, int H, int W)
{
    Tensor t(C, H, W);
    size_t nbytes = C * H * W * sizeof(float);

    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "Cannot open: " << path << "\n";
        exit(1);
    }
    f.read(reinterpret_cast<char*>(t.data.data()), nbytes);
    return t;
}
