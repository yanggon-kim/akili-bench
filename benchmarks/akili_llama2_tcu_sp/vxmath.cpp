#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>

#include <rvfloats.h>
#include <tensor.h>
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

struct SparseWeightKey {
    const float* ptr;
    uint32_t rows;
    uint32_t cols;

    bool operator==(const SparseWeightKey& other) const {
        return ptr == other.ptr && rows == other.rows && cols == other.cols;
    }
};

struct SparseWeightKeyHash {
    size_t operator()(const SparseWeightKey& key) const {
        size_t seed = std::hash<const void*>{}(key.ptr);
        seed ^= std::hash<uint32_t>{}(key.rows) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<uint32_t>{}(key.cols) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};

struct QkvWeightKey {
    const float* wq;
    const float* wk;
    const float* wv;
    uint32_t dim;
    uint32_t kv_dim;

    bool operator==(const QkvWeightKey& other) const {
        return wq == other.wq && wk == other.wk && wv == other.wv
            && dim == other.dim && kv_dim == other.kv_dim;
    }
};

struct QkvWeightKeyHash {
    size_t operator()(const QkvWeightKey& key) const {
        size_t seed = std::hash<const void*>{}(key.wq);
        seed ^= std::hash<const void*>{}(key.wk) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<const void*>{}(key.wv) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<uint32_t>{}(key.dim) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<uint32_t>{}(key.kv_dim) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};

struct FfnWeightKey {
    const float* w1;
    const float* w3;
    uint32_t input_dim;
    uint32_t hidden_dim;

    bool operator==(const FfnWeightKey& other) const {
        return w1 == other.w1 && w3 == other.w3
            && input_dim == other.input_dim && hidden_dim == other.hidden_dim;
    }
};

struct FfnWeightKeyHash {
    size_t operator()(const FfnWeightKey& key) const {
        size_t seed = std::hash<const void*>{}(key.w1);
        seed ^= std::hash<const void*>{}(key.w3) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<uint32_t>{}(key.input_dim) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<uint32_t>{}(key.hidden_dim) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};

struct SparseWeight {
    uint32_t logical_rows = 0;
    uint32_t logical_cols = 0;
    uint32_t padded_rows = 0;
    uint32_t padded_cols = 0;
    vx_buffer_h data_buffer = nullptr;
    vx_buffer_h meta_buffer = nullptr;
    uint64_t data_addr = 0;
    uint64_t meta_addr = 0;
};

static uint64_t num_threads = 0;
static vx_device_h device = nullptr;
static vx_buffer_h matmul_B_buffer = nullptr;
static vx_buffer_h matmul_C_buffer = nullptr;
static vx_buffer_h matmul_krnl_buffer = nullptr;
static vx_buffer_h matmul_args_buffer = nullptr;
static matmul_kernel_args_t matmul_kernel_arg = {};
static bool kernel_loaded = false;
static size_t cached_B_bytes = 0;
static size_t cached_C_bytes = 0;
static const char* kernel = "kernel.vxbin";

static std::unordered_map<SparseWeightKey, std::unique_ptr<SparseWeight>, SparseWeightKeyHash> sparse_weight_cache;
static std::unordered_map<QkvWeightKey, std::unique_ptr<SparseWeight>, QkvWeightKeyHash> qkv_weight_cache;
static std::unordered_map<FfnWeightKey, std::unique_ptr<SparseWeight>, FfnWeightKeyHash> ffn_weight_cache;

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

static void release_sparse_weight(SparseWeight* weight) {
    if (weight->data_buffer) {
        vx_mem_free(weight->data_buffer);
        weight->data_buffer = nullptr;
    }
    if (weight->meta_buffer) {
        vx_mem_free(weight->meta_buffer);
        weight->meta_buffer = nullptr;
    }
}

static void release_sparse_cache() {
    for (auto& entry : sparse_weight_cache) {
        release_sparse_weight(entry.second.get());
    }
    sparse_weight_cache.clear();

    for (auto& entry : qkv_weight_cache) {
        release_sparse_weight(entry.second.get());
    }
    qkv_weight_cache.clear();

    for (auto& entry : ffn_weight_cache) {
        release_sparse_weight(entry.second.get());
    }
    ffn_weight_cache.clear();
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

static void pack_metadata(std::vector<uint32_t>& h_meta,
                          const std::vector<uint8_t>& masks,
                          uint32_t M,
                          uint32_t K) {
    constexpr uint32_t i_ratio = cfg::rtl_i_ratio;
    constexpr uint32_t tc_k = cfg::tcK;
    constexpr uint32_t tc_m = cfg::tcM;
    constexpr uint32_t meta_row_w = tc_k * 2 * i_ratio;
    constexpr uint32_t mcols = cfg::meta_cols;
    constexpr uint32_t half_k_steps = cfg::k_steps / 2;
    constexpr uint32_t pd = cfg::m_steps * (cfg::k_steps / 2);
    constexpr uint32_t cols_per_load = (NUM_THREADS >= pd) ? (NUM_THREADS / pd) : 1;
    constexpr uint32_t banks_per_store = (NUM_THREADS < pd) ? NUM_THREADS : pd;
    constexpr uint32_t stores_per_col = (pd + NUM_THREADS - 1) / NUM_THREADS;
    constexpr uint32_t num_meta_loads = (pd * mcols + NUM_THREADS - 1) / NUM_THREADS;
    constexpr uint32_t per_k_tile_words = num_meta_loads * NUM_THREADS;

    uint32_t tile_k_elem = cfg::tileK;
    uint32_t num_groups_per_row = K / 4;
    uint32_t elts_per_sparse_step = tile_k_elem / half_k_steps;

    uint32_t num_tile_rows = M / cfg::tileM;
    uint32_t num_k_tiles = K / cfg::tileK;

    h_meta.assign(num_tile_rows * num_k_tiles * per_k_tile_words, 0);

    for (uint32_t tr = 0; tr < num_tile_rows; ++tr) {
        for (uint32_t kt = 0; kt < num_k_tiles; ++kt) {
            uint32_t section_base = (tr * num_k_tiles + kt) * per_k_tile_words;

            for (uint32_t sm = 0; sm < cfg::m_steps; ++sm) {
                for (uint32_t sk = 0; sk < half_k_steps; ++sk) {
                    uint32_t sram_row = sm * half_k_steps + sk;

                    for (uint32_t i = 0; i < tc_m; ++i) {
                        uint32_t physical_row = tr * cfg::tileM + sm * tc_m + i;
                        uint32_t k_elem_start = kt * tile_k_elem + sk * elts_per_sparse_step;

                        for (uint32_t e = 0; e < elts_per_sparse_step; ++e) {
                            uint32_t global_elt = k_elem_start + e;
                            uint32_t global_group = global_elt / 4;
                            uint32_t pos_in_group = global_elt % 4;
                            uint8_t mask = masks[physical_row * num_groups_per_row + global_group];

                            if (mask & (1u << pos_in_group)) {
                                uint32_t k_reg = e / (2 * i_ratio);
                                uint32_t pos_in_k = e % (2 * i_ratio);
                                uint32_t meta_bit;
                                if (pos_in_k < i_ratio) {
                                    meta_bit = k_reg * i_ratio + pos_in_k;
                                } else {
                                    meta_bit = (tc_k + k_reg) * i_ratio + (pos_in_k - i_ratio);
                                }
                                uint32_t block_bit = i * meta_row_w + meta_bit;
                                uint32_t word_idx = block_bit / 32;
                                uint32_t bit_idx = block_bit % 32;
                                uint32_t store_in_col = sram_row / banks_per_store;
                                uint32_t thread_in_store = sram_row % banks_per_store;
                                uint32_t flat_store = word_idx * stores_per_col + store_in_col;
                                uint32_t load_idx = flat_store / cols_per_load;
                                uint32_t store_in_load = flat_store % cols_per_load;
                                uint32_t meta_idx = load_idx * NUM_THREADS + store_in_load * banks_per_store + thread_in_store;
                                h_meta[section_base + meta_idx] |= (1u << bit_idx);
                            }
                        }
                    }
                }
            }
        }
    }
}

static std::unique_ptr<SparseWeight> prepare_sparse_weight(const float* dense_weight,
                                                           uint32_t rows,
                                                           uint32_t cols) {
    auto weight = std::make_unique<SparseWeight>();
    weight->logical_rows = rows;
    weight->logical_cols = cols;
    weight->padded_rows = round_up(rows, cfg::tileM);
    weight->padded_cols = round_up(cols, cfg::tileK);

    size_t dense_size = static_cast<size_t>(weight->padded_rows) * weight->padded_cols;
    size_t sparse_size = dense_size / 2;

    std::vector<itype_t> dense_fp16(dense_size, 0);
    std::vector<itype_t> compressed_fp16(sparse_size, 0);
    std::vector<uint8_t> masks;
    std::vector<uint32_t> packed_meta;

    pack_row_major_fp16(dense_fp16, dense_weight, rows, cols, weight->padded_cols);

    if (!vt::prune_2to4_matrix<vt::fp16>(dense_fp16.data(), weight->padded_rows, weight->padded_cols)) {
        std::cerr << "prune_2to4_matrix failed for sparse llama weight" << std::endl;
        exit(-1);
    }

    if (!vt::compress_2to4_matrix<vt::fp16>(compressed_fp16.data(),
                                            dense_fp16.data(),
                                            masks,
                                            weight->padded_rows,
                                            weight->padded_cols)) {
        std::cerr << "compress_2to4_matrix failed for sparse llama weight" << std::endl;
        exit(-1);
    }

    pack_metadata(packed_meta, masks, weight->padded_rows, weight->padded_cols);

    RT_CHECK(vx_mem_alloc(device, compressed_fp16.size() * sizeof(itype_t), VX_MEM_READ, &weight->data_buffer));
    RT_CHECK(vx_mem_address(weight->data_buffer, &weight->data_addr));
    RT_CHECK(vx_copy_to_dev(weight->data_buffer,
                            compressed_fp16.data(),
                            0,
                            compressed_fp16.size() * sizeof(itype_t)));

    RT_CHECK(vx_mem_alloc(device, packed_meta.size() * sizeof(uint32_t), VX_MEM_READ, &weight->meta_buffer));
    RT_CHECK(vx_mem_address(weight->meta_buffer, &weight->meta_addr));
    RT_CHECK(vx_copy_to_dev(weight->meta_buffer,
                            packed_meta.data(),
                            0,
                            packed_meta.size() * sizeof(uint32_t)));

    return weight;
}

static SparseWeight* get_or_create_sparse_weight(const float* weight_ptr,
                                                 uint32_t rows,
                                                 uint32_t cols) {
    SparseWeightKey key{weight_ptr, rows, cols};
    auto it = sparse_weight_cache.find(key);
    if (it != sparse_weight_cache.end()) {
        return it->second.get();
    }

    auto weight = prepare_sparse_weight(weight_ptr, rows, cols);
    auto* raw_weight = weight.get();
    sparse_weight_cache.emplace(key, std::move(weight));
    return raw_weight;
}

static SparseWeight* get_or_create_qkv_weight(const float* wq,
                                              const float* wk,
                                              const float* wv,
                                              uint32_t dim,
                                              uint32_t kv_dim) {
    QkvWeightKey key{wq, wk, wv, dim, kv_dim};
    auto it = qkv_weight_cache.find(key);
    if (it != qkv_weight_cache.end()) {
        return it->second.get();
    }

    uint32_t rows = dim + 2 * kv_dim;
    std::vector<float> concat(static_cast<size_t>(rows) * dim);
    std::memcpy(concat.data(), wq, static_cast<size_t>(dim) * dim * sizeof(float));
    std::memcpy(concat.data() + static_cast<size_t>(dim) * dim,
                wk,
                static_cast<size_t>(kv_dim) * dim * sizeof(float));
    std::memcpy(concat.data() + static_cast<size_t>(dim + kv_dim) * dim,
                wv,
                static_cast<size_t>(kv_dim) * dim * sizeof(float));

    auto weight = prepare_sparse_weight(concat.data(), rows, dim);
    auto* raw_weight = weight.get();
    qkv_weight_cache.emplace(key, std::move(weight));
    return raw_weight;
}

static SparseWeight* get_or_create_ffn_weight(const float* w1,
                                              const float* w3,
                                              uint32_t input_dim,
                                              uint32_t hidden_dim) {
    FfnWeightKey key{w1, w3, input_dim, hidden_dim};
    auto it = ffn_weight_cache.find(key);
    if (it != ffn_weight_cache.end()) {
        return it->second.get();
    }

    uint32_t rows = 2 * hidden_dim;
    std::vector<float> concat(static_cast<size_t>(rows) * input_dim);
    std::memcpy(concat.data(), w1, static_cast<size_t>(hidden_dim) * input_dim * sizeof(float));
    std::memcpy(concat.data() + static_cast<size_t>(hidden_dim) * input_dim,
                w3,
                static_cast<size_t>(hidden_dim) * input_dim * sizeof(float));

    auto weight = prepare_sparse_weight(concat.data(), rows, input_dim);
    auto* raw_weight = weight.get();
    ffn_weight_cache.emplace(key, std::move(weight));
    return raw_weight;
}

static void run_sparse_matmul(float* C,
                              SparseWeight* weight,
                              const float* B,
                              uint32_t M,
                              uint32_t N,
                              uint32_t K) {
    uint32_t padded_N = round_up(N, cfg::tileN);
    size_t sizeB = static_cast<size_t>(weight->padded_cols) * padded_N;
    size_t sizeC = static_cast<size_t>(weight->padded_rows) * padded_N;

    std::vector<itype_t> host_B(sizeB, 0);
    std::vector<otype_t> host_C(sizeC, 0.0f);

    pack_row_major_fp16(host_B, B, K, N, padded_N);

    matmul_kernel_arg.M = weight->padded_rows;
    matmul_kernel_arg.N = padded_N;
    matmul_kernel_arg.K = weight->padded_cols;
    matmul_kernel_arg.A_addr = weight->data_addr;
    matmul_kernel_arg.meta_sp_addr = weight->meta_addr;

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

    RT_CHECK(vx_copy_to_dev(matmul_B_buffer, host_B.data(), 0, sizeB * sizeof(itype_t)));
    RT_CHECK(vx_copy_to_dev(matmul_args_buffer, &matmul_kernel_arg, 0, sizeof(matmul_kernel_args_t)));

    uint32_t grid_dim[2] = {padded_N / cfg::tileN, weight->padded_rows / cfg::tileM};
    uint32_t block_dim[2] = {NUM_THREADS, 1};

    RT_CHECK(vx_start_g(device, matmul_krnl_buffer, matmul_args_buffer, 2, grid_dim, block_dim, 0));
    RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
    RT_CHECK(vx_copy_from_dev(host_C.data(), matmul_C_buffer, 0, sizeC * sizeof(otype_t)));

    for (uint32_t row = 0; row < M; ++row) {
        std::memcpy(C + row * N,
                    host_C.data() + row * padded_N,
                    static_cast<size_t>(N) * sizeof(float));
    }
}

void vx_init() {
    std::cout << "open device connection" << std::endl;
    RT_CHECK(vx_dev_open(&device));

    uint64_t isa_flags = 0;
    RT_CHECK(vx_dev_caps(device, VX_CAPS_ISA_FLAGS, &isa_flags));
    if ((isa_flags & VX_ISA_EXT_TCU) == 0) {
        std::cerr << "TCU extension not supported by this Vortex build" << std::endl;
        exit(-1);
    }

    RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_THREADS, &num_threads));
    if (num_threads != NUM_THREADS) {
        std::cerr << "TCU kernel expects NUM_THREADS=" << NUM_THREADS
                  << ", but device reports " << num_threads << std::endl;
        exit(-1);
    }
}

void matmul_cleanup() {
    if (!device) {
        return;
    }

    release_sparse_cache();

    if (matmul_B_buffer) {
        vx_mem_free(matmul_B_buffer);
        matmul_B_buffer = nullptr;
    }
    if (matmul_C_buffer) {
        vx_mem_free(matmul_C_buffer);
        matmul_C_buffer = nullptr;
    }
    if (matmul_krnl_buffer) {
        vx_mem_free(matmul_krnl_buffer);
        matmul_krnl_buffer = nullptr;
    }
    if (matmul_args_buffer) {
        vx_mem_free(matmul_args_buffer);
        matmul_args_buffer = nullptr;
    }

    cached_B_bytes = 0;
    cached_C_bytes = 0;
    kernel_loaded = false;

    vx_dev_close(device);
    device = nullptr;
}

void vx_matmul(float* C, float* A, float* B, int M, int N, int K) {
    auto* sparse_weight = get_or_create_sparse_weight(A, M, K);
    run_sparse_matmul(C, sparse_weight, B, M, N, K);
}

void vx_matmul_qkv_batch(float* Q, float* K, float* V,
                         float* input,
                         float* Wq, float* Wk, float* Wv,
                         int dim, int kv_dim) {
    auto* sparse_weight = get_or_create_qkv_weight(Wq, Wk, Wv, dim, kv_dim);
    int total_output_dim = dim + 2 * kv_dim;
    std::vector<float> output_concat(total_output_dim);

    run_sparse_matmul(output_concat.data(), sparse_weight, input, total_output_dim, 1, dim);

    std::memcpy(Q, output_concat.data(), static_cast<size_t>(dim) * sizeof(float));
    std::memcpy(K, output_concat.data() + dim, static_cast<size_t>(kv_dim) * sizeof(float));
    std::memcpy(V, output_concat.data() + dim + kv_dim, static_cast<size_t>(kv_dim) * sizeof(float));
}

void vx_matmul_ffn_batch(float* out1, float* out2, float* input,
                         float* W1, float* W3,
                         int input_dim, int hidden_dim) {
    auto* sparse_weight = get_or_create_ffn_weight(W1, W3, input_dim, hidden_dim);
    int total_output_dim = 2 * hidden_dim;
    std::vector<float> output_concat(total_output_dim);

    run_sparse_matmul(output_concat.data(), sparse_weight, input, total_output_dim, 1, input_dim);

    std::memcpy(out1, output_concat.data(), static_cast<size_t>(hidden_dim) * sizeof(float));
    std::memcpy(out2, output_concat.data() + hidden_dim, static_cast<size_t>(hidden_dim) * sizeof(float));
}
