#include "cnn_pipeline.h"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>

//
// Paths to compiled kernels
//
static const char* CONV_KERNEL  = "layers/conv_kernel.vxbin";
static const char* POOL_KERNEL  = "layers/pool_kernel.vxbin";
static const char* RELU_KERNEL  = "layers/relu_kernel.vxbin";

//
// Paths to weights exported from Python
//
static const char* CONV_W_FILE  = "weights/conv1_w.bin";
static const char* CONV_B_FILE  = "weights/conv1_b.bin";

static const char* FC_W_FILE    = "weights/fc_w.bin";
static const char* FC_B_FILE    = "weights/fc_b.bin";

//
// Path to test image exported from Python
//
static const char* IMAGE_FILE   = "images/test_image.bin";

//
// ------------------------------------------------------------
// Load a 28×28 Fashion-MNIST image from .bin (float32)
// ------------------------------------------------------------
Tensor load_fashion_image(const std::string& path) {
    // C=1, H=28, W=28
    Tensor t(1, 28, 28);

    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "Cannot open: " << path << "\n";
        exit(1);
    }
    f.read(reinterpret_cast<char*>(t.data.data()), 28*28*sizeof(float));
    return t;
}

//
// ------------------------------------------------------------
// main()
// ------------------------------------------------------------
int main() {

    std::cout << "----- Vortex *Accelerated!* Fashion-MNIST CNN -----" << std::endl;

    //
    // Open Vortex device
    //
    VortexDevice vx;
    std::cout << "Opened Vortex device.\n";

    //
    // Load image
    //
    std::cout << "Loading image...\n";
    Tensor image = load_fashion_image(IMAGE_FILE);

    //
    // Load weights
    //
    std::cout << "Loading weights...\n";
    std::vector<float> conv_W = load_bin_vector(CONV_W_FILE);   // shape 8×1×3×3
    std::vector<float> conv_B = load_bin_vector(CONV_B_FILE);   // shape 8

    std::vector<float> fc_W = load_bin_vector(FC_W_FILE);       // shape 10×1352
    std::vector<float> fc_B = load_bin_vector(FC_B_FILE);       // shape 10

    //
    // Forward pass
    //

    // ------------------ Conv ------------------
    std::cout << "Running Conv2D...\n";
    Tensor conv_out = conv2d_gpu(
        vx,
        image,
        conv_W,
        conv_B,
        /*C_out=*/8,
        /*K=*/3,
        /*padding=*/0,   // valid padding
        /*stride=*/1,
        CONV_KERNEL
    );

    // conv_out shape = (8, 26, 26)
    std::cout << "Conv output: "
              << conv_out.C << "x"
              << conv_out.H << "x"
              << conv_out.W << "\n";

    // ------------------ Pool ------------------
    std::cout << "Running MaxPool2D...\n";
    Tensor pool_out = maxpool2d_gpu(
        vx,
        conv_out,
        /*pool_size=*/2,
        POOL_KERNEL
    );

    // pool_out = (8, 13, 13)
    std::cout << "Pool output: "
              << pool_out.C << "x"
              << pool_out.H << "x"
              << pool_out.W << "\n";

    // ------------------ ReLU ------------------
    std::cout << "Running ReLU...\n";
    Tensor relu_out = relu_gpu(
        vx,
        pool_out,
        RELU_KERNEL
    );

    // ------------------ FC ------------------
    std::cout << "Running Fully Connected...\n";
    std::vector<float> logits = fc_cpu(
        relu_out,
        fc_W,
        fc_B
    );

    std::vector<float> probs = softmax(logits);
    int pred = argmax(probs);

    //
    // Print results
    //
    std::cout << "Prediction: " << pred << "\n";
    std::cout << "Probabilities: ";
    for (float p : probs)
        std::cout << p << " ";
    std::cout << "\n";

    std::cout << "----- Done -----\n";
    return 0;
}
