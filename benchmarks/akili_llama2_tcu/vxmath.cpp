#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include <rvfloats.h>
#include <tensor_cfg.h>
#include <vortex.h>

#include "common.h"

#define RT_CHECK(_expr)                                         \
    do {                                                        \
        int _ret = (_expr);                                     \
        if (0 == _ret)                                          \
            break;                                              \
        printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);\
        exit(-1);                                               \
    } while (false)

namespace vt = vortex::tensor;
using cfg = vt::wmma_config_t<NUM_THREADS, vt::fp16, vt::fp32>;
using itype_t = typename vt::fp16::dtype;
using otype_t = typename vt::fp32::dtype;

template <typename To, typename From>
static To bit_cast(const From& src) {
    union {
        From from;
        To to;
    } cast = {src};
    return cast.to;
}

static uint32_t round_up(uint32_t value, uint32_t multiple) {
    return ((value + multiple - 1) / multiple) * multiple;
}

static uint16_t float_to_fp16(float value) {
    return rv_ftoh_s(bit_cast<uint32_t>(value), 0, nullptr);
}

static uint64_t num_cores = 0;
static uint64_t num_warps = 0;
static uint64_t num_threads = 0;

static vx_device_h device = nullptr;
static vx_buffer_h matmul_A_buffer = nullptr;
static vx_buffer_h matmul_B_buffer = nullptr;
static vx_buffer_h matmul_C_buffer = nullptr;
static vx_buffer_h matmul_krnl_buffer = nullptr;
static vx_buffer_h matmul_args_buffer = nullptr;
static matmul_kernel_args_t matmul_kernel_arg = {};
static bool kernel_loaded = false;

static size_t cached_A_bytes = 0;
static size_t cached_B_bytes = 0;
static size_t cached_C_bytes = 0;

static const char* kernel = "kernel.vxbin";

static void ensure_buffer(vx_buffer_h* buffer,
                          size_t* cached_bytes,
                          size_t bytes,
                          int flags,
                          uint64_t* address) {
    if (!*buffer || bytes > *cached_bytes) {
        if (*buffer) {
            vx_mem_free(*buffer);
        }
        RT_CHECK(vx_mem_alloc(device, bytes, flags, buffer));
        *cached_bytes = bytes;
    }
    RT_CHECK(vx_mem_address(*buffer, address));
}

void vx_init() {
    std::cout << "open device connection" << std::endl;
    RT_CHECK(vx_dev_open(&device));
    RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_CORES, &num_cores));
    RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_WARPS, &num_warps));
    RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_THREADS, &num_threads));

    uint64_t isa_flags = 0;
    RT_CHECK(vx_dev_caps(device, VX_CAPS_ISA_FLAGS, &isa_flags));
    if ((isa_flags & VX_ISA_EXT_TCU) == 0) {
        std::cerr << "TCU extension not supported by this Vortex build" << std::endl;
        exit(-1);
    }

    if (num_threads != NUM_THREADS) {
        std::cerr << "TCU kernel expects NUM_THREADS=" << NUM_THREADS
                  << ", but device reports " << num_threads << std::endl;
        exit(-1);
    }
}

void matmul_cleanup() {
    if (device) {
        if (matmul_A_buffer) vx_mem_free(matmul_A_buffer);
        if (matmul_B_buffer) vx_mem_free(matmul_B_buffer);
        if (matmul_C_buffer) vx_mem_free(matmul_C_buffer);
        if (matmul_krnl_buffer) vx_mem_free(matmul_krnl_buffer);
        if (matmul_args_buffer) vx_mem_free(matmul_args_buffer);

        matmul_A_buffer = nullptr;
        matmul_B_buffer = nullptr;
        matmul_C_buffer = nullptr;
        matmul_krnl_buffer = nullptr;
        matmul_args_buffer = nullptr;
        cached_A_bytes = 0;
        cached_B_bytes = 0;
        cached_C_bytes = 0;
        kernel_loaded = false;

        vx_dev_close(device);
        device = nullptr;
    }
}

static void pack_row_major_fp16(std::vector<itype_t>& dst,
                                const float* src,
                                uint32_t rows,
                                uint32_t cols,
                                uint32_t padded_cols) {
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
            dst[r * padded_cols + c] = float_to_fp16(src[r * cols + c]);
        }
    }
}

static void vx_matmul_tcu(float* C, const float* A, const float* B, int M, int N, int K) {
    uint32_t padded_M = round_up(M, cfg::tileM);
    uint32_t padded_N = round_up(N, cfg::tileN);
    uint32_t padded_K = round_up(K, cfg::tileK);

    size_t sizeA = static_cast<size_t>(padded_M) * padded_K;
    size_t sizeB = static_cast<size_t>(padded_K) * padded_N;
    size_t sizeC = static_cast<size_t>(padded_M) * padded_N;

    std::vector<itype_t> host_A(sizeA, 0);
    std::vector<itype_t> host_B(sizeB, 0);
    std::vector<otype_t> host_C(sizeC, 0.0f);

    pack_row_major_fp16(host_A, A, M, K, padded_K);
    pack_row_major_fp16(host_B, B, K, N, padded_N);

    matmul_kernel_arg.M = padded_M;
    matmul_kernel_arg.N = padded_N;
    matmul_kernel_arg.K = padded_K;

    ensure_buffer(&matmul_A_buffer, &cached_A_bytes, sizeA * sizeof(itype_t),
                  VX_MEM_READ, &matmul_kernel_arg.A_addr);
    ensure_buffer(&matmul_B_buffer, &cached_B_bytes, sizeB * sizeof(itype_t),
                  VX_MEM_READ, &matmul_kernel_arg.B_addr);
    ensure_buffer(&matmul_C_buffer, &cached_C_bytes, sizeC * sizeof(otype_t),
                  VX_MEM_WRITE, &matmul_kernel_arg.C_addr);

    if (!kernel_loaded) {
        RT_CHECK(vx_upload_kernel_file(device, kernel, &matmul_krnl_buffer));
        kernel_loaded = true;
    }

    if (!matmul_args_buffer) {
        RT_CHECK(vx_mem_alloc(device, sizeof(matmul_kernel_args_t), VX_MEM_READ, &matmul_args_buffer));
    }

    RT_CHECK(vx_copy_to_dev(matmul_A_buffer, host_A.data(), 0, sizeA * sizeof(itype_t)));
    RT_CHECK(vx_copy_to_dev(matmul_B_buffer, host_B.data(), 0, sizeB * sizeof(itype_t)));
    RT_CHECK(vx_copy_to_dev(matmul_args_buffer, &matmul_kernel_arg, 0, sizeof(matmul_kernel_args_t)));

    uint32_t grid_dim[2] = {padded_N / cfg::tileN, padded_M / cfg::tileM};
    uint32_t block_dim[2] = {NUM_THREADS, 1};

    RT_CHECK(vx_start_g(device, matmul_krnl_buffer, matmul_args_buffer, 2,
                        grid_dim, block_dim, 0));
    RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
    RT_CHECK(vx_copy_from_dev(host_C.data(), matmul_C_buffer, 0, sizeC * sizeof(otype_t)));

    for (int row = 0; row < M; ++row) {
        std::memcpy(C + row * N,
                    host_C.data() + row * padded_N,
                    static_cast<size_t>(N) * sizeof(float));
    }
}

void vx_matmul(float* C, float* A, float* B, int M, int N, int K) {
    vx_matmul_tcu(C, A, B, M, N, K);
}

void vx_matmul_qkv_batch(float* Q, float* K, float* V,
                         float* input,
                         float* Wq, float* Wk, float* Wv,
                         int dim, int kv_dim) {
    int total_output_dim = dim + 2 * kv_dim;
    int input_dim = dim;

    float* W_concat = static_cast<float*>(std::malloc(
        static_cast<size_t>(total_output_dim) * input_dim * sizeof(float)));
    float* output_concat = static_cast<float*>(std::malloc(
        static_cast<size_t>(total_output_dim) * sizeof(float)));

    std::memcpy(W_concat, Wq, static_cast<size_t>(dim) * input_dim * sizeof(float));
    std::memcpy(W_concat + dim * input_dim, Wk,
                static_cast<size_t>(kv_dim) * input_dim * sizeof(float));
    std::memcpy(W_concat + (dim + kv_dim) * input_dim, Wv,
                static_cast<size_t>(kv_dim) * input_dim * sizeof(float));

    vx_matmul_tcu(output_concat, W_concat, input, total_output_dim, 1, input_dim);

    std::memcpy(Q, output_concat, static_cast<size_t>(dim) * sizeof(float));
    std::memcpy(K, output_concat + dim, static_cast<size_t>(kv_dim) * sizeof(float));
    std::memcpy(V, output_concat + dim + kv_dim, static_cast<size_t>(kv_dim) * sizeof(float));

    std::free(W_concat);
    std::free(output_concat);
}

void vx_matmul_ffn_batch(float* out1, float* out2, float* input,
                         float* W1, float* W3,
                         int input_dim, int hidden_dim) {
    int total_output_dim = 2 * hidden_dim;

    float* W_concat = static_cast<float*>(std::malloc(
        static_cast<size_t>(total_output_dim) * input_dim * sizeof(float)));
    float* output_concat = static_cast<float*>(std::malloc(
        static_cast<size_t>(total_output_dim) * sizeof(float)));

    std::memcpy(W_concat, W1, static_cast<size_t>(hidden_dim) * input_dim * sizeof(float));
    std::memcpy(W_concat + hidden_dim * input_dim, W3,
                static_cast<size_t>(hidden_dim) * input_dim * sizeof(float));

    vx_matmul_tcu(output_concat, W_concat, input, total_output_dim, 1, input_dim);

    std::memcpy(out1, output_concat, static_cast<size_t>(hidden_dim) * sizeof(float));
    std::memcpy(out2, output_concat + hidden_dim, static_cast<size_t>(hidden_dim) * sizeof(float));

    std::free(W_concat);
    std::free(output_concat);
}
