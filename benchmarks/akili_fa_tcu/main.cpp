// akili_fa_tcu / akili_fa_tcu_sp host: fused WMMA FlashAttention-2 forward,
// O = softmax(Q Kᵀ) V with Q, K, V [N x FA_D], checked against a CPU
// reference on the fp16-rounded (and, under FA_SPARSE, 2:4-pruned) inputs.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <vortex.h>
#include "wgmma_dxa_host.h"
#include "common.h"

#ifndef VX_CFG_NUM_TCU_LANES
#define VX_CFG_NUM_TCU_LANES VX_CFG_NUM_THREADS
#endif

#include <tensor_cfg.h>
#include <tensor_sp.h>
namespace vt = vortex::tensor;
using cfg = vt::wmma_config_t<VX_CFG_NUM_TCU_LANES, vt::fp16, vt::fp32>;

#define RT_CHECK(_expr)                                                     \
  do {                                                                      \
    int _ret = _expr;                                                       \
    if (0 == _ret) break;                                                   \
    printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);                \
    cleanup();                                                              \
    exit(-1);                                                               \
  } while (false)

constexpr uint32_t kWarpsPerCta = FA_BR / cfg::tileM;
#if defined(FA_SPARSE) && !defined(FA_META_GLOBAL)
constexpr uint32_t kMetaBytes = kWarpsPerCta * (FA_D / cfg::tileK) * cfg::num_meta_loads * VX_CFG_NUM_THREADS * 4;
#else
constexpr uint32_t kMetaBytes = 0;
#endif
constexpr uint32_t kLmemBytes = kWarpsPerCta * cfg::tileM * (FA_BC * 4 + FA_BC * 2 + FA_D * 4) + kMetaBytes;

static vx_device_h device = nullptr;
static vx_buffer_h Q_buf = nullptr, K_buf = nullptr, V_buf = nullptr, O_buf = nullptr, M_buf = nullptr;

static void cleanup() {
  if (device) {
    if (Q_buf) vx_mem_free(Q_buf);
    if (K_buf) vx_mem_free(K_buf);
    if (V_buf) vx_mem_free(V_buf);
    if (O_buf) vx_mem_free(O_buf);
    if (M_buf) vx_mem_free(M_buf);
    vx_dev_close(device);
  }
}

static float frand() { return 10.0f * (float(rand()) / RAND_MAX) - 5.0f; }

static uint16_t f2h_host(float x) {
  uint32_t bits;
  std::memcpy(&bits, &x, sizeof(bits));
  uint16_t sign = (uint16_t)((bits >> 16) & 0x8000);
  uint16_t mant = (uint16_t)((bits >> 13) & 0x03FF);
  int32_t ex = (int32_t)((bits >> 23) & 0xFF) - 127 + 15;
  if (ex >= 31) return (uint16_t)(sign | 0x7C00);
  if (ex <= 0) return sign;
  return (uint16_t)(sign | ((uint16_t)ex << 10) | mant);
}

static float h2f_host(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000) << 16;
  uint32_t ex = (h >> 10) & 0x1F, mant = h & 0x3FF;
  uint32_t bits = ex == 0 ? sign : (sign | ((ex - 15 + 127) << 23) | (mant << 13));
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

#ifdef FA_SPARSE
// WMMA sparse-A metadata, [M/tileM][K/tileK][meta_stride words]; same packing
// as tests/regression/sgemm_tcu_sp.
static void pack_metadata(std::vector<uint32_t>& h_meta,
                          const std::vector<uint8_t>& masks,
                          uint32_t M, uint32_t K) {
  constexpr uint32_t I_RATIO = cfg::rtl_i_ratio;
  constexpr uint32_t TC_K = cfg::tcK;
  constexpr uint32_t TC_M = cfg::tcM;
  constexpr uint32_t meta_row_w = TC_K * 2 * I_RATIO;
  constexpr uint32_t mcols = cfg::meta_cols;
  constexpr uint32_t half_k_steps = cfg::k_steps / 2;
  constexpr uint32_t PD = cfg::m_steps * (cfg::k_steps / 2);
  constexpr uint32_t banks_per_store = (VX_CFG_NUM_THREADS < PD) ? VX_CFG_NUM_THREADS : PD;
  constexpr uint32_t stores_per_col = (PD + VX_CFG_NUM_THREADS - 1) / VX_CFG_NUM_THREADS;
  constexpr uint32_t cols_per_load = (VX_CFG_NUM_THREADS >= PD) ? (VX_CFG_NUM_THREADS / PD) : 1;
  constexpr uint32_t num_meta_loads = (PD * mcols + VX_CFG_NUM_THREADS - 1) / VX_CFG_NUM_THREADS;
  constexpr uint32_t per_k_tile_words = num_meta_loads * VX_CFG_NUM_THREADS;

  uint32_t tileK_elem = cfg::tileK;
  uint32_t num_groups_per_row = K / 4;
  uint32_t elts_per_sparse_step = tileK_elem / half_k_steps;
  uint32_t num_tile_rows = M / cfg::tileM;
  uint32_t num_k_tiles = K / cfg::tileK;

  h_meta.assign(num_tile_rows * num_k_tiles * per_k_tile_words, 0);
  for (uint32_t tr = 0; tr < num_tile_rows; ++tr) {
    for (uint32_t kt = 0; kt < num_k_tiles; ++kt) {
      uint32_t section_base = (tr * num_k_tiles + kt) * per_k_tile_words;
      for (uint32_t sm = 0; sm < cfg::m_steps; ++sm) {
        for (uint32_t sk = 0; sk < half_k_steps; ++sk) {
          uint32_t sram_row = sm * half_k_steps + sk;
          for (uint32_t i = 0; i < TC_M; ++i) {
            uint32_t physical_row = tr * cfg::tileM + sm * TC_M + i;
            uint32_t k_elem_start = kt * tileK_elem + sk * elts_per_sparse_step;
            for (uint32_t e = 0; e < elts_per_sparse_step; ++e) {
              uint32_t global_elt = k_elem_start + e;
              uint8_t mask = masks[physical_row * num_groups_per_row + global_elt / 4];
              if (!(mask & (1u << (global_elt % 4)))) continue;
              uint32_t k_reg = e / (2 * I_RATIO);
              uint32_t pos_in_k = e % (2 * I_RATIO);
              uint32_t meta_bit = pos_in_k < I_RATIO
                  ? k_reg * I_RATIO + pos_in_k
                  : (TC_K + k_reg) * I_RATIO + (pos_in_k - I_RATIO);
              uint32_t block_bit = i * meta_row_w + meta_bit;
              uint32_t word_idx = block_bit / 32;
              uint32_t store_in_col = sram_row / banks_per_store;
              uint32_t thread_in_store = sram_row % banks_per_store;
              uint32_t flat_store = word_idx * stores_per_col + store_in_col;
              uint32_t load_idx = flat_store / cols_per_load;
              uint32_t store_in_load = flat_store % cols_per_load;
              uint32_t meta_idx = load_idx * VX_CFG_NUM_THREADS
                                + store_in_load * banks_per_store + thread_in_store;
              h_meta[section_base + meta_idx] |= 1u << (block_bit % 32);
            }
          }
        }
      }
    }
  }
}
#endif

static void attention_cpu(std::vector<float>& O, const std::vector<float>& Q,
                          const std::vector<float>& K, const std::vector<float>& V,
                          uint32_t N, uint32_t d) {
  std::vector<double> s(N);
  for (uint32_t i = 0; i < N; ++i) {
    double mx = -1e300;
    for (uint32_t j = 0; j < N; ++j) {
      double acc = 0;
      for (uint32_t k = 0; k < d; ++k) acc += (double)Q[i * d + k] * K[j * d + k];
      s[j] = acc;
      mx = acc > mx ? acc : mx;
    }
    double sum = 0;
    for (uint32_t j = 0; j < N; ++j) { s[j] = std::exp(s[j] - mx); sum += s[j]; }
    for (uint32_t k = 0; k < d; ++k) {
      double acc = 0;
      for (uint32_t j = 0; j < N; ++j) acc += s[j] * V[j * d + k];
      O[i * d + k] = (float)(acc / sum);
    }
  }
}

int main(int argc, char** argv) {
  uint32_t N = 512;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "-n") && i + 1 < argc) N = (uint32_t)atoi(argv[++i]);
  }
  if (N % FA_BR || N % FA_BC) {
    printf("Error: N=%u must be a multiple of %u\n", N, FA_BR);
    return -1;
  }
  const uint32_t d = FA_D;
#ifdef FA_SPARSE
  const char* name = "akili_fa_tcu_sp";
#else
  const char* name = "akili_fa_tcu";
#endif
  printf("%s: N=%u d=%u Br=%u Bc=%u lmem=%u\n", name, N, d, FA_BR, FA_BC, kLmemBytes);

  RT_CHECK(vx_dev_open(&device));

  srand(42);
  std::vector<float> h_Q(N * d), h_K(N * d), h_V(N * d);
  std::vector<uint16_t> q16(N * d), k16(N * d), v16(N * d);
  for (uint32_t i = 0; i < N * d; ++i) {
    q16[i] = f2h_host(frand());
    k16[i] = f2h_host(frand()); h_K[i] = h2f_host(k16[i]);
    v16[i] = f2h_host(frand()); h_V[i] = h2f_host(v16[i]);
  }

  kernel_arg_t arg = {};
  std::vector<uint16_t> q_dev = q16;
#ifdef FA_SPARSE
  if (!vt::prune_2to4_matrix<vt::fp16>(q16.data(), N, d)) {
    printf("prune_2to4_matrix(Q) failed\n"); cleanup(); return -1;
  }
  q_dev.assign(N * (d / 2), 0);
  std::vector<uint8_t> masks;
  if (!vt::compress_2to4_matrix<vt::fp16>(q_dev.data(), q16.data(), masks, N, d)) {
    printf("compress_2to4_matrix(Q) failed\n"); cleanup(); return -1;
  }
  std::vector<uint32_t> h_meta;
  pack_metadata(h_meta, masks, N, d);
  uint32_t meta_bytes = (uint32_t)(h_meta.size() * sizeof(uint32_t));
  RT_CHECK(vx_mem_alloc(device, meta_bytes, VX_MEM_READ, &M_buf));
  RT_CHECK(vx_mem_address(M_buf, &arg.meta_addr));
  RT_CHECK(vx_copy_to_dev(M_buf, h_meta.data(), 0, meta_bytes));
#endif
  for (uint32_t i = 0; i < N * d; ++i) h_Q[i] = h2f_host(q16[i]);

  uint32_t in_bytes = N * d * sizeof(uint16_t);
  uint32_t q_bytes = (uint32_t)(q_dev.size() * sizeof(uint16_t));
  RT_CHECK(vx_mem_alloc(device, q_bytes, VX_MEM_READ, &Q_buf));
  RT_CHECK(vx_mem_alloc(device, in_bytes, VX_MEM_READ, &K_buf));
  RT_CHECK(vx_mem_alloc(device, in_bytes, VX_MEM_READ, &V_buf));
  RT_CHECK(vx_mem_alloc(device, N * d * sizeof(float), VX_MEM_WRITE, &O_buf));
  RT_CHECK(vx_mem_address(Q_buf, &arg.Q_addr));
  RT_CHECK(vx_mem_address(K_buf, &arg.K_addr));
  RT_CHECK(vx_mem_address(V_buf, &arg.V_addr));
  RT_CHECK(vx_mem_address(O_buf, &arg.O_addr));
  RT_CHECK(vx_copy_to_dev(Q_buf, q_dev.data(), 0, q_bytes));
  RT_CHECK(vx_copy_to_dev(K_buf, k16.data(), 0, in_bytes));
  RT_CHECK(vx_copy_to_dev(V_buf, v16.data(), 0, in_bytes));
  arg.N = N;

  uint32_t grid[2] = {N / FA_BR, 1};
  uint32_t block[2] = {kWarpsPerCta * VX_CFG_NUM_THREADS, 1};
  auto t0 = std::chrono::high_resolution_clock::now();
  RT_CHECK(akili_start_g(device, &arg, sizeof(arg), 1, grid, block, kLmemBytes));
  auto t1 = std::chrono::high_resolution_clock::now();
  printf("FLASH: %ld us (grid=%u)\n",
         (long)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count(), grid[0]);

  std::vector<float> h_O(N * d);
  RT_CHECK(vx_copy_from_dev(h_O.data(), O_buf, 0, N * d * sizeof(float)));

  std::vector<float> ref(N * d);
  attention_cpu(ref, h_Q, h_K, h_V, N, d);
  const float atol = 2e-2f, rtol = 2e-2f;
  int errors = 0;
  for (uint32_t i = 0; i < N * d; ++i) {
    if (std::fabs(h_O[i] - ref[i]) > atol + rtol * std::fabs(ref[i])) {
      if (errors < 10)
        printf("*** O error: [%u] expected=%f, actual=%f\n", i, ref[i], h_O[i]);
      ++errors;
    }
  }

  cleanup();
  if (errors) {
    printf("Found %d / %u errors!\nFAILED!\n", errors, N * d);
    return errors;
  }
  printf("PASSED!\n");
  return 0;
}
