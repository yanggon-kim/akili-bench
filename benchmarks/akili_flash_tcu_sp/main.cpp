// akili_attn_tcu_sp — Sparse TCU attention (2:4 on both Q and P).
//
// Three kernel launches per run:
//   Stage 1: sparse Q·Kᵀ  (host pre-prunes Q on the device upload path)
//   Stage 2: SIMT softmax on fp32 S
//   Stage 3: sparse P·V   (host prunes P after the softmax readback,
//                          uploads compressed P + metadata, kernel PV
//                          uses the sparse MMA path)

#include <iostream>
#include <unistd.h>
#include <string.h>
#include <cstring>
#include <vector>
#include <chrono>
#include <vortex.h>
#include <cmath>
#include <algorithm>
#include <stdio.h>
#include "common.h"

#ifndef NUM_THREADS
#define NUM_THREADS 8
#endif
#ifndef NUM_TCU_LANES
#define NUM_TCU_LANES NUM_THREADS
#endif

#include <tensor_cfg.h>
#include <tensor.h>  // prune_2to4_matrix, compress_2to4_matrix
namespace vt = vortex::tensor;
using cfg = vt::wmma_config_t<NUM_TCU_LANES, vt::fp16, vt::fp32>;
static constexpr uint32_t TM = cfg::tileM;
static constexpr uint32_t TN = cfg::tileN;
static constexpr uint32_t TK = cfg::tileK;

// Tile col-major B into contiguous tileK×tileN blocks for cache-friendly
// sparse fragB loads. Block(n_tile, k_tile) stored col-major with stride=tileK.
static void tile_B_colmajor(std::vector<uint16_t>& B_tiled,
                            const std::vector<uint16_t>& B_cm,
                            uint32_t K_walk, uint32_t N_cols) {
  uint32_t num_k_tiles = K_walk / TK;
  uint32_t num_n_tiles = N_cols / TN;
  B_tiled.resize(K_walk * N_cols);
  uint32_t offset = 0;
  for (uint32_t nt = 0; nt < num_n_tiles; ++nt)
    for (uint32_t kt = 0; kt < num_k_tiles; ++kt)
      for (uint32_t col = 0; col < TN; ++col)
        for (uint32_t row = 0; row < TK; ++row)
          B_tiled[offset++] = B_cm[(nt * TN + col) * K_walk + kt * TK + row];
}

// -----------------------------------------------------------------------------
// pack_metadata — lifted verbatim from tests/regression/sgemm_tcu_sp/main.cpp.
// Parameterized by (M, K) so we can reuse it for both the Q-prune pass
// (M=N, K=d) and the P-prune pass (M=N, K=N).
// -----------------------------------------------------------------------------
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
  constexpr uint32_t banks_per_store = (NUM_THREADS < PD) ? NUM_THREADS : PD;
  constexpr uint32_t stores_per_col = (PD + NUM_THREADS - 1) / NUM_THREADS;
  constexpr uint32_t cols_per_load = (NUM_THREADS >= PD) ? (NUM_THREADS / PD) : 1;
  constexpr uint32_t num_meta_loads = (PD * mcols + NUM_THREADS - 1) / NUM_THREADS;
  constexpr uint32_t per_k_tile_words = num_meta_loads * NUM_THREADS;

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
              uint32_t global_group = global_elt / 4;
              uint32_t pos_in_group = global_elt % 4;
              uint8_t mask = masks[physical_row * num_groups_per_row + global_group];
              if (mask & (1u << pos_in_group)) {
                uint32_t k_reg = e / (2 * I_RATIO);
                uint32_t pos_in_k = e % (2 * I_RATIO);
                uint32_t meta_bit;
                if (pos_in_k < I_RATIO) {
                  meta_bit = k_reg * I_RATIO + pos_in_k;
                } else {
                  meta_bit = (TC_K + k_reg) * I_RATIO + (pos_in_k - I_RATIO);
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

#define RT_CHECK(_expr)                                         \
   do {                                                         \
     int _ret = _expr;                                          \
     if (0 == _ret) break;                                      \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
     cleanup();                                                 \
     exit(-1);                                                  \
   } while (false)

static float frand() { return 10.0f * (float(rand()) / RAND_MAX) - 5.0f; }

static inline uint16_t f2h_host(float x) {
  uint32_t bits;
  std::memcpy(&bits, &x, sizeof(bits));
  uint16_t sign = (bits >> 16) & 0x8000;
  uint16_t mant = (bits >> 13) & 0x03FF;
  int32_t  exp  = (int32_t)((bits >> 23) & 0xFF) - 127 + 15;
  if (exp >= 31) return sign | 0x7C00;
  if (exp <= 0)  return sign;
  return sign | ((uint16_t)exp << 10) | mant;
}
static inline float h2f_host(uint16_t x) {
  uint32_t sign = (uint32_t)(x & 0x8000) << 16;
  uint32_t exp  = (x >> 10) & 0x1F;
  uint32_t mant = (x & 0x3FF);
  uint32_t bits;
  if (exp == 0)       bits = sign;
  else if (exp == 31) bits = sign | 0x7F800000 | (mant << 13);
  else                bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
  float f; std::memcpy(&f, &bits, sizeof(f)); return f;
}

static bool compare(float a, float b, float atol, float rtol,
                    int index, int& errors, const char* tag) {
  auto diff = std::fabs(a - b);
  auto limit = atol + rtol * std::fabs(b);
  if (diff > limit) {
    if (errors < 20)
      printf("*** %s error: [%d] expected=%f, actual=%f\n", tag, index, a, b);
    return false;
  }
  return true;
}

static void matmul_cpu(float* out, const float* A, const float* B,
                       uint32_t M, uint32_t N, uint32_t K) {
  for (uint32_t row = 0; row < M; ++row)
    for (uint32_t col = 0; col < N; ++col) {
      double sum = 0;
      for (uint32_t e = 0; e < K; ++e) sum += (double)A[row*K+e] * (double)B[e*N+col];
      out[row*N+col] = (float)sum;
    }
}
static void softmax_cpu(float* out, const float* A, uint32_t M, uint32_t N) {
  for (uint32_t row = 0; row < M; ++row) {
    float max_val = A[row*N];
    for (uint32_t col = 1; col < N; ++col)
      max_val = std::max(max_val, A[row*N + col]);
    float exp_sum = 0;
    for (uint32_t col = 0; col < N; ++col) {
      out[row*N + col] = std::exp(A[row*N + col] - max_val);
      exp_sum += out[row*N + col];
    }
    for (uint32_t col = 0; col < N; ++col) out[row*N + col] /= exp_sum;
  }
}

// ---------------------------------------------------------------------------

const char* kernel_file = "kernel.vxbin";
uint32_t N_req = 16;
uint32_t d_req = 16;

vx_device_h device = nullptr;
vx_buffer_h Qsp_buffer   = nullptr;
vx_buffer_h K_fp16_buf   = nullptr;
vx_buffer_h V_fp16_buf   = nullptr;
vx_buffer_h S_buffer     = nullptr;
vx_buffer_h P_fp32_buf   = nullptr;
vx_buffer_h P_sp_buffer  = nullptr;
vx_buffer_h O_buffer     = nullptr;
vx_buffer_h meta_Q_buf   = nullptr;
vx_buffer_h meta_P_buf   = nullptr;
vx_buffer_h cycles_buffer = nullptr;
vx_buffer_h krnl_buffer  = nullptr;
vx_buffer_h args_buffer  = nullptr;
kernel_arg_t kernel_arg  = {};

static void show_usage() {
  std::cout << "akili_attn_tcu_sp — Sparse TCU attention benchmark" << std::endl;
  std::cout << "Usage: [-n N] [-d D] [-k kernel_file]" << std::endl;
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "n:d:k:h")) != -1) {
    switch (c) {
      case 'n': N_req = atoi(optarg); break;
      case 'd': d_req = atoi(optarg); break;
      case 'k': kernel_file = optarg; break;
      case 'h': show_usage(); exit(0);
      default:  show_usage(); exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    if (Qsp_buffer)  vx_mem_free(Qsp_buffer);
    if (K_fp16_buf)  vx_mem_free(K_fp16_buf);
    if (V_fp16_buf)  vx_mem_free(V_fp16_buf);
    if (S_buffer)    vx_mem_free(S_buffer);
    if (P_fp32_buf)  vx_mem_free(P_fp32_buf);
    if (P_sp_buffer) vx_mem_free(P_sp_buffer);
    if (O_buffer)    vx_mem_free(O_buffer);
    if (meta_Q_buf)  vx_mem_free(meta_Q_buf);
    if (meta_P_buf)  vx_mem_free(meta_P_buf);
    if (cycles_buffer) vx_mem_free(cycles_buffer);
    if (krnl_buffer) vx_mem_free(krnl_buffer);
    if (args_buffer) vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

static void read_back_cycles(const char* stage_tag, uint32_t num_blocks) {
  std::vector<uint32_t> h_cycles(num_blocks, 0);
  vx_copy_from_dev(h_cycles.data(), cycles_buffer, 0, num_blocks * sizeof(uint32_t));
  uint32_t max_cyc = 0;
  for (auto c : h_cycles) if (c > max_cyc) max_cyc = c;
  printf("KCYC[%s,nt=%u]: %u\n", stage_tag, (unsigned)NUM_THREADS, max_cyc);
}

int main(int argc, char* argv[]) {
  parse_args(argc, argv);
  std::srand(50);

  RT_CHECK(vx_dev_open(&device));
  uint64_t NT_caps = 0;
  vx_dev_caps(device, VX_CAPS_NUM_THREADS, &NT_caps);
  if ((uint32_t)NT_caps != NUM_THREADS) {
    printf("Error: device NUM_THREADS=%u but built with NUM_THREADS=%u\n",
           (unsigned)NT_caps, (unsigned)NUM_THREADS);
    cleanup();
    return -1;
  }

  auto round_up = [](uint32_t v, uint32_t m) { return ((v + m - 1) / m) * m; };
  uint32_t n_align = std::max({TM, TN, TK});
  uint32_t d_align = std::max(TK, TN);
  uint32_t N = round_up(N_req, n_align);
  uint32_t d = round_up(d_req, d_align);
  std::cout << "akili_attn_tcu_sp — padded N=" << N << " (req " << N_req
            << "), d=" << d << " (req " << d_req << "), "
            << "TM=" << TM << " TN=" << TN << " TK=" << TK << std::endl;

  std::vector<float> h_Q(N * d, 0.0f);
  std::vector<float> h_K(d * N, 0.0f);
  std::vector<float> h_V(N * d, 0.0f);
  for (uint32_t r = 0; r < N_req; ++r)
    for (uint32_t c = 0; c < d_req; ++c)
      h_Q[r * d + c] = frand();
  for (uint32_t r = 0; r < d_req; ++r)
    for (uint32_t c = 0; c < N_req; ++c)
      h_K[r * N + c] = frand();
  for (uint32_t r = 0; r < N_req; ++r)
    for (uint32_t c = 0; c < d_req; ++c)
      h_V[r * d + c] = frand();

  // ---- Prune + compress + pack metadata for Q ----
  std::vector<uint16_t> h_Q_fp16(N * d);
  for (uint32_t i = 0; i < N * d; ++i) h_Q_fp16[i] = f2h_host(h_Q[i]);
  if (!vt::prune_2to4_matrix<vt::fp16>(h_Q_fp16.data(), N, d)) {
    printf("prune_2to4_matrix(Q) failed\n"); cleanup(); return -1;
  }
  for (uint32_t i = 0; i < N * d; ++i) h_Q[i] = h2f_host(h_Q_fp16[i]);

  std::vector<uint16_t> h_Q_compressed(N * (d / 2));
  std::vector<uint8_t>  sparse_masks_Q;
  if (!vt::compress_2to4_matrix<vt::fp16>(h_Q_compressed.data(),
                                          h_Q_fp16.data(),
                                          sparse_masks_Q, N, d)) {
    printf("compress_2to4_matrix(Q) failed\n"); cleanup(); return -1;
  }
  std::vector<uint32_t> h_meta_Q;
  pack_metadata(h_meta_Q, sparse_masks_Q, N, d);

  // ---- Pack K (col-major) and V (col-major) as fp16, then tile for cache ----
  std::vector<uint16_t> h_K_cm(N * d);
  std::vector<uint16_t> h_V_cm(N * d);
  for (uint32_t n = 0; n < N; ++n)
    for (uint32_t k = 0; k < d; ++k)
      h_K_cm[n * d + k] = f2h_host(h_K[k * N + n]);
  for (uint32_t d_idx = 0; d_idx < d; ++d_idx)
    for (uint32_t n = 0; n < N; ++n)
      h_V_cm[d_idx * N + n] = f2h_host(h_V[n * d + d_idx]);

  // Tile K and V into contiguous tileK×tileN blocks.
  // QK stage: B=K, K_walk=d, N_cols=N. PV stage: B=V, K_walk=N, N_cols=d.
  std::vector<uint16_t> h_K_tiled, h_V_tiled;
  tile_B_colmajor(h_K_tiled, h_K_cm, d, N);
  tile_B_colmajor(h_V_tiled, h_V_cm, N, d);

  // ---- Allocate device buffers ----
  uint32_t qsp_bytes   = (uint32_t)h_Q_compressed.size() * sizeof(uint16_t);
  uint32_t q_meta_bytes = (uint32_t)h_meta_Q.size() * sizeof(uint32_t);
  uint32_t in_fp16_bytes = N * d * sizeof(uint16_t);
  uint32_t s_fp32_bytes = N * N * sizeof(float);
  uint32_t psp_bytes    = N * (N / 2) * sizeof(uint16_t);
  uint32_t out_bytes    = N * d * sizeof(float);

  RT_CHECK(vx_mem_alloc(device, qsp_bytes,      VX_MEM_READ_WRITE, &Qsp_buffer));
  RT_CHECK(vx_mem_address(Qsp_buffer, &kernel_arg.Q_addr));
  RT_CHECK(vx_mem_alloc(device, in_fp16_bytes,  VX_MEM_READ_WRITE, &K_fp16_buf));
  RT_CHECK(vx_mem_address(K_fp16_buf, &kernel_arg.K_addr));
  RT_CHECK(vx_mem_alloc(device, in_fp16_bytes,  VX_MEM_READ_WRITE, &V_fp16_buf));
  RT_CHECK(vx_mem_address(V_fp16_buf, &kernel_arg.V_addr));
  RT_CHECK(vx_mem_alloc(device, s_fp32_bytes,   VX_MEM_READ_WRITE, &S_buffer));
  RT_CHECK(vx_mem_address(S_buffer, &kernel_arg.S_addr));
  RT_CHECK(vx_mem_alloc(device, s_fp32_bytes,   VX_MEM_READ_WRITE, &P_fp32_buf));
  uint64_t P_fp32_addr = 0;
  RT_CHECK(vx_mem_address(P_fp32_buf, &P_fp32_addr));
  RT_CHECK(vx_mem_alloc(device, psp_bytes,      VX_MEM_READ_WRITE, &P_sp_buffer));
  uint64_t P_sp_addr = 0;
  RT_CHECK(vx_mem_address(P_sp_buffer, &P_sp_addr));
  RT_CHECK(vx_mem_alloc(device, out_bytes,      VX_MEM_READ_WRITE, &O_buffer));
  RT_CHECK(vx_mem_address(O_buffer, &kernel_arg.O_addr));
  RT_CHECK(vx_mem_alloc(device, q_meta_bytes,   VX_MEM_READ_WRITE, &meta_Q_buf));
  RT_CHECK(vx_mem_address(meta_Q_buf, &kernel_arg.meta_Q_addr));
  // meta_P is sized at use-time after the softmax readback — reserve 64 KB
  // which covers every (M=N, K=N) shape up to ~N=256.
  RT_CHECK(vx_mem_alloc(device, 64 * 1024,      VX_MEM_READ_WRITE, &meta_P_buf));
  RT_CHECK(vx_mem_address(meta_P_buf, &kernel_arg.meta_P_addr));

  kernel_arg.N = N;
  kernel_arg.d = d;

  RT_CHECK(vx_copy_to_dev(Qsp_buffer, h_Q_compressed.data(), 0, qsp_bytes));
  RT_CHECK(vx_copy_to_dev(meta_Q_buf, h_meta_Q.data(),       0, q_meta_bytes));
  // Use plain col-major (sgemm_tcu_sp pattern).
  RT_CHECK(vx_copy_to_dev(K_fp16_buf, h_K_cm.data(),         0, in_fp16_bytes));
  RT_CHECK(vx_copy_to_dev(V_fp16_buf, h_V_cm.data(),         0, in_fp16_bytes));

  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
  RT_CHECK(vx_mem_alloc(device, sizeof(kernel_arg_t), VX_MEM_READ_WRITE, &args_buffer));

  uint32_t qk_blocks = (N / TN) * (N / TM);
  uint32_t sm_blocks = (N + NUM_THREADS - 1) / NUM_THREADS;
  uint32_t pv_blocks = (d / TN) * (N / TM);
  uint32_t max_blocks = qk_blocks;
  if (sm_blocks > max_blocks) max_blocks = sm_blocks;
  if (pv_blocks > max_blocks) max_blocks = pv_blocks;
  RT_CHECK(vx_mem_alloc(device, max_blocks * sizeof(uint32_t),
                        VX_MEM_READ_WRITE, &cycles_buffer));
  RT_CHECK(vx_mem_address(cycles_buffer, &kernel_arg.cycles_addr));

  int errors = 0;
  const float atol = 2e-2f;
  const float rtol = 2e-2f;

  uint32_t block_dim[2] = {NUM_THREADS, 1};

  // ---------------- Stage 1: sparse QK ----------------
  std::cout << "=== Stage 1: S = Q @ K^T (sparse TCU) ===" << std::endl;
  kernel_arg.kernel_id = KID_QK_SPARSE;
  RT_CHECK(vx_copy_to_dev(args_buffer, &kernel_arg, 0, sizeof(kernel_arg_t)));
  {
    uint32_t grid_dim[2] = {N / TN, N / TM};
    RT_CHECK(vx_start_g(device, krnl_buffer, args_buffer, 2, grid_dim, block_dim, 0));
  }
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
  read_back_cycles("QK", qk_blocks);

  std::vector<float> h_S(N * N, 0.0f);
  RT_CHECK(vx_copy_from_dev(h_S.data(), S_buffer, 0, s_fp32_bytes));
  {
    std::vector<float> h_S_ref(N * N, 0.0f);
    matmul_cpu(h_S_ref.data(), h_Q.data(), h_K.data(), N, N, d);
    for (uint32_t r = 0; r < N_req; ++r)
      for (uint32_t c = 0; c < N_req; ++c) {
        int idx = r * N + c;
        if (!compare(h_S_ref[idx], h_S[idx], atol, rtol, idx, errors, "S"))
          ++errors;
      }
  }
  if (errors) {
    std::cout << "S stage found " << errors << " errors, continuing" << std::endl;
    errors = 0;
  }

  for (uint32_t r = 0; r < N; ++r)
    for (uint32_t c = 0; c < N; ++c)
      if (r >= N_req || c >= N_req) h_S[r * N + c] = -1e9f;
  RT_CHECK(vx_copy_to_dev(S_buffer, h_S.data(), 0, s_fp32_bytes));

  // ---------------- Stage 2: SIMT softmax ----------------
  std::cout << "=== Stage 2: P = softmax(S) ===" << std::endl;
  kernel_arg.P_addr    = P_fp32_addr;
  kernel_arg.kernel_id = KID_SOFTMAX;
  RT_CHECK(vx_copy_to_dev(args_buffer, &kernel_arg, 0, sizeof(kernel_arg_t)));
  {
    uint32_t grid_dim[2] = {sm_blocks, 1};
    RT_CHECK(vx_start_g(device, krnl_buffer, args_buffer, 1, grid_dim, block_dim, 0));
  }
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
  read_back_cycles("SM", sm_blocks);

  std::vector<float> h_P(N * N);
  RT_CHECK(vx_copy_from_dev(h_P.data(), P_fp32_buf, 0, s_fp32_bytes));
  {
    std::vector<float> h_P_ref(N * N);
    softmax_cpu(h_P_ref.data(), h_S.data(), N, N);
    for (uint32_t r = 0; r < N_req; ++r)
      for (uint32_t c = 0; c < N_req; ++c) {
        int idx = r * N + c;
        if (!compare(h_P_ref[idx], h_P[idx], 1e-5f, 1e-5f, idx, errors, "P"))
          ++errors;
      }
  }
  if (errors) {
    std::cout << "P stage found " << errors << " errors, continuing" << std::endl;
    errors = 0;
  }

  // ---------------- Stage 3: sparse PV ----------------
  // Host prunes P 2:4, compresses, packs metadata, uploads.
  std::cout << "=== Stage 3: O = P @ V (sparse TCU) ===" << std::endl;
  std::vector<uint16_t> h_P_fp16(N * N);
  for (uint32_t i = 0; i < N * N; ++i) h_P_fp16[i] = f2h_host(h_P[i]);
  if (!vt::prune_2to4_matrix<vt::fp16>(h_P_fp16.data(), N, N)) {
    printf("prune_2to4_matrix(P) failed\n"); cleanup(); return -1;
  }
  for (uint32_t i = 0; i < N * N; ++i) h_P[i] = h2f_host(h_P_fp16[i]);

  std::vector<uint16_t> h_P_compressed(N * (N / 2));
  std::vector<uint8_t>  sparse_masks_P;
  if (!vt::compress_2to4_matrix<vt::fp16>(h_P_compressed.data(),
                                          h_P_fp16.data(),
                                          sparse_masks_P, N, N)) {
    printf("compress_2to4_matrix(P) failed\n"); cleanup(); return -1;
  }
  std::vector<uint32_t> h_meta_P;
  pack_metadata(h_meta_P, sparse_masks_P, N, N);

  RT_CHECK(vx_copy_to_dev(P_sp_buffer, h_P_compressed.data(), 0,
                          (uint32_t)h_P_compressed.size() * sizeof(uint16_t)));
  RT_CHECK(vx_copy_to_dev(meta_P_buf,  h_meta_P.data(),        0,
                          (uint32_t)h_meta_P.size() * sizeof(uint32_t)));

  kernel_arg.P_addr    = P_sp_addr;
  kernel_arg.kernel_id = KID_PV_SPARSE;
  RT_CHECK(vx_copy_to_dev(args_buffer, &kernel_arg, 0, sizeof(kernel_arg_t)));
  {
    uint32_t grid_dim[2] = {d / TN, N / TM};
    RT_CHECK(vx_start_g(device, krnl_buffer, args_buffer, 2, grid_dim, block_dim, 0));
  }
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
  read_back_cycles("PV", pv_blocks);

  std::vector<float> h_O(N * d);
  RT_CHECK(vx_copy_from_dev(h_O.data(), O_buffer, 0, out_bytes));
  {
    std::vector<float> h_O_ref(N * d, 0.0f);
    matmul_cpu(h_O_ref.data(), h_P.data(), h_V.data(), N, d, N);
    for (uint32_t r = 0; r < N_req; ++r)
      for (uint32_t c = 0; c < d_req; ++c) {
        int idx = r * d + c;
        if (!compare(h_O_ref[idx], h_O[idx], atol, rtol, idx, errors, "O"))
          ++errors;
      }
  }

  cleanup();

  if (errors != 0) {
    std::cout << "Found " << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return errors;
  }
  std::cout << "PASSED!" << std::endl;
  return 0;
}
