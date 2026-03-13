#include "cnn_pipeline.h"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <limits>
#include <iomanip>
#include <algorithm>

// -------------------- Debug helpers --------------------

void print_tensor_stats(const Tensor& t, const std::string& name) {
    float mn = std::numeric_limits<float>::infinity();
    float mx = -std::numeric_limits<float>::infinity();
    double sum = 0.0;

    for (float v : t.data) {
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        sum += v;
    }
    double mean = sum / (double)t.data.size();

    std::cout << name << " shape: "
              << t.C << "x" << t.H << "x" << t.W << "\n";
    std::cout << name << " min=" << mn
              << ", max=" << mx
              << ", mean=" << mean << "\n";
}

void print_first_pixel_channels(const Tensor& t, const std::string& name) {
    std::cout << name << "[y=0,x=0,c=0..7] =";
    int maxC = std::min(t.C, 8);
    for (int c = 0; c < maxC; ++c) {
        float v = t.data[c * t.H * t.W + 0 * t.W + 0];
        std::cout << " " << std::setprecision(6) << v;
    }
    std::cout << "\n";
}

static void print_logits(const std::vector<float>& logits) {
    std::cout << "logits =";
    for (float v : logits)
        std::cout << " " << std::setprecision(6) << v;
    std::cout << "\n";
}

static void print_probs(const std::vector<float>& probs) {
    std::cout << "Probabilities: ";
    for (float p : probs)
        std::cout << p << " ";
    std::cout << "\n";
}

// -------------------- Files --------------------

static const char* CONV_KERNEL  = "layers/conv_kernel.vxbin";
static const char* POOL_KERNEL  = "layers/pool_kernel.vxbin";
static const char* RELU_KERNEL  = "layers/relu_kernel.vxbin";

static const char* CONV_W_FILE  = "weights/conv1_w.bin";
static const char* CONV_B_FILE  = "weights/conv1_b.bin";

static const char* FC_W_FILE    = "weights/fc_w.bin";
static const char* FC_B_FILE    = "weights/fc_b.bin";

// -------------------- Multi-image config --------------------

static const int  NUM_TEST    = 100;   // run 0..NUM_TEST-1
static const int  DEBUG_IDX   = 0;     // which image prints full debug
static const bool PRINT_EACH  = true;  // print one-line per image

static std::string img_path(int i)   { return "images/img_"   + std::to_string(i) + ".bin"; }
static std::string label_path(int i) { return "images/label_" + std::to_string(i) + ".txt"; }

// -------------------- IO --------------------

int load_label(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        std::cerr << "Cannot open: " << path << "\n";
        exit(1);
    }
    int y;
    f >> y;
    return y;
}

Tensor load_fashion_image(const std::string& path) {
    Tensor t(1, 28, 28);
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "Cannot open: " << path << "\n";
        exit(1);
    }
    f.read(reinterpret_cast<char*>(t.data.data()), 28 * 28 * sizeof(float));
    return t;
}

// -------------------- main --------------------

int main() {
    std::cout << "----- Vortex Fashion-MNIST CNN -----" << std::endl;

    static const char* class_names[10] = {
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    };

    // Open device once
    VortexDevice vx;
    std::cout << "Opened Vortex device.\n";

    // Load weights once
    std::cout << "Loading weights...\n";
    std::vector<float> conv_W = load_bin_vector(CONV_W_FILE);
    std::vector<float> conv_B = load_bin_vector(CONV_B_FILE);
    std::vector<float> fc_W   = load_bin_vector(FC_W_FILE);
    std::vector<float> fc_B   = load_bin_vector(FC_B_FILE);

    int correct = 0;

    for (int i = 0; i < NUM_TEST; ++i) {
        Tensor image = load_fashion_image(img_path(i));
        int label = load_label(label_path(i));

        // ------------------ Conv ------------------
        Tensor conv_out = conv2d_gpu(
            vx,
            image,
            conv_W,
            conv_B,
            /*C_out=*/8,
            /*K=*/3,
            /*padding=*/0,
            /*stride=*/1,
            CONV_KERNEL
        );

        // ------------------ Pool ------------------
        Tensor pool_out = maxpool2d_gpu(
            vx,
            conv_out,
            /*pool_size=*/2,
            POOL_KERNEL
        );

        // ------------------ ReLU ------------------
        Tensor relu_out = relu_gpu(
            vx,
            pool_out,
            RELU_KERNEL
        );

        // Tensor relu_out = pool_out;   // BYPASS ReLU kernel for sanity test


        // ------------------ FC ------------------
        std::vector<float> logits = fc_cpu(relu_out, fc_W, fc_B);
        std::vector<float> probs  = softmax(logits);
        int pred = argmax(probs);

        bool ok = (pred == label);
        if (ok) correct++;

        if (PRINT_EACH) {
            std::cout << "img " << i
                      << " | gt=" << label << " (" << class_names[label] << ")"
                      << " | pred=" << pred << " (" << class_names[pred] << ")"
                      << " | " << (ok ? "CORRECT" : "WRONG")
                      << "\n";
        }

        // Keep your debug statements (only for one chosen image)
        if (i == DEBUG_IDX) {
            std::cout << "\n--- DEBUG (img " << i << ") ---\n";
            std::cout << "Conv output: " << conv_out.C << "x" << conv_out.H << "x" << conv_out.W << "\n";
            print_tensor_stats(conv_out, "conv_out");
            print_first_pixel_channels(conv_out, "conv_out");

            std::cout << "Pool output: " << pool_out.C << "x" << pool_out.H << "x" << pool_out.W << "\n";
            print_tensor_stats(pool_out, "pool_out");
            print_first_pixel_channels(pool_out, "pool_out");

            print_tensor_stats(relu_out, "relu_out");
            print_first_pixel_channels(relu_out, "relu_out");

            print_logits(logits);

            std::cout << "Ground truth: " << label << " (" << class_names[label] << ")\n";
            std::cout << "Prediction : " << pred  << " (" << class_names[pred]  << ")\n";
            std::cout << "Result: " << (ok ? "CORRECT" : "WRONG") << "\n";
            std::cout << "Prediction: " << pred << "\n";
            print_probs(probs);
            std::cout << "--- END DEBUG ---\n\n";
        }
    }

    std::cout << "\n----- Summary -----\n";
    std::cout << "Tested " << NUM_TEST << " images\n";
    std::cout << "Correct: " << correct << "\n";
    std::cout << "Accuracy: " << (100.0 * correct / NUM_TEST) << "%\n";
    std::cout << "----- Done -----\n";

    return 0;
}
