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
#include <tensor.h>  // vortex::tensor::prune_2to4_matrix, compress_2to4_matrix
namespace vt = vortex::tensor;
using cfg = vt::wmma_config_t<NUM_TCU_LANES, vt::fp16, vt::fp32>;
static constexpr uint32_t TM = cfg::tileM;
static constexpr uint32_t TN = cfg::tileN;
static constexpr uint32_t TK = cfg::tileK;  // already in element units

// Pack per-group 2:4 masks into the SRAM layout expected by VX_tcu_meta.
// Copied from tests/regression/sgemm_tcu_sp/main.cpp — identical formula.
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
  constexpr uint32_t cols_per_load = (NUM_THREADS >= PD) ? (NUM_THREADS / PD) : 1;
  constexpr uint32_t banks_per_store = (NUM_THREADS < PD) ? NUM_THREADS : PD;
  constexpr uint32_t stores_per_col = (PD + NUM_THREADS - 1) / NUM_THREADS;
  constexpr uint32_t num_meta_loads = (PD * mcols + NUM_THREADS - 1) / NUM_THREADS;
  constexpr uint32_t per_k_tile_words = num_meta_loads * NUM_THREADS;

  uint32_t tileK_elem = cfg::tileK;
  uint32_t KS = K;
  uint32_t num_groups_per_row = KS / 4;
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
     if (0 == _ret)                                             \
       break;                                                   \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
     cleanup();                                                 \
     exit(-1);                                                  \
   } while (false)

///////////////////////////////////////////////////////////////////////////////

static float frand() {
  return 10.0f * (float(rand()) / RAND_MAX) - 5.0f;
}

// Relative-tolerance compare, with mode-dependent tolerance.
static bool compare(float a, float b, float atol, float rtol,
                    int index, int& errors, const char* tag) {
  auto diff = std::fabs(a - b);
  auto limit = atol + rtol * std::fabs(b);
  if (diff > limit) {
    if (errors < 20) {
      printf("*** %s error: [%d] expected=%f, actual=%f\n", tag, index, a, b);
    }
    return false;
  }
  return true;
}

static void matmul_cpu(float* out, const float* A, const float* B,
                       uint32_t M, uint32_t N, uint32_t K) {
  for (uint32_t row = 0; row < M; ++row) {
    for (uint32_t col = 0; col < N; ++col) {
      double sum = 0;
      for (uint32_t e = 0; e < K; ++e) sum += (double)A[row*K+e] * (double)B[e*N+col];
      out[row*N+col] = (float)sum;
    }
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

// Reference implementation for the full fused attention: O = softmax(Q @ K^T) @ V.
// Uses the layout K[N × d] (flash's fused-path convention — each row is one
// token's key vector). Q and K dotproducts are computed as Q[row] · K[col].
static void attention_cpu_fused(float* out, const float* Q, const float* K,
                                 const float* V, uint32_t N, uint32_t d) {
  std::vector<float> scores(N);
  std::vector<float> probs(N);
  for (uint32_t i = 0; i < N; ++i) {
    for (uint32_t j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (uint32_t k = 0; k < d; ++k) sum += Q[i * d + k] * K[j * d + k];
      scores[j] = sum;
    }
    float m = scores[0];
    for (uint32_t j = 1; j < N; ++j) m = std::max(m, scores[j]);
    float es = 0.0f;
    for (uint32_t j = 0; j < N; ++j) { probs[j] = std::exp(scores[j] - m); es += probs[j]; }
    for (uint32_t j = 0; j < N; ++j) probs[j] /= es;
    for (uint32_t k = 0; k < d; ++k) {
      float o = 0.0f;
      for (uint32_t j = 0; j < N; ++j) o += probs[j] * V[j * d + k];
      out[i * d + k] = o;
    }
  }
}

// ---------------------------------------------------------------------------

const char* kernel_file = "kernel.vxbin";
uint32_t N_req = 16;       // requested sequence length
uint32_t d_req = 16;       // requested head dim
uint32_t mode  = MODE_SIMT; // 0=SIMT, 1=dense TCU

vx_device_h device = nullptr;
vx_buffer_h Q_buffer = nullptr;
vx_buffer_h K_buffer = nullptr;
vx_buffer_h S_buffer = nullptr;
vx_buffer_h P_buffer = nullptr;
vx_buffer_h V_buffer = nullptr;
vx_buffer_h O_buffer = nullptr;
// TCU fp16 device buffers (allocated only when mode != SIMT)
vx_buffer_h Q_fp16_buffer    = nullptr;
vx_buffer_h K_fp16_cm_buffer = nullptr;
vx_buffer_h V_fp16_cm_buffer = nullptr;
vx_buffer_h P_fp16_buffer    = nullptr;
vx_buffer_h P_sp_buffer      = nullptr;   // compressed fp16 P for sparse PV
vx_buffer_h meta_P_buffer    = nullptr;   // packed 2:4 metadata for P
vx_buffer_h Qsp_buffer = nullptr;    // compressed fp16 Q for sparse mode
vx_buffer_h meta_Q_buffer = nullptr; // sparse metadata for Q
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};
uint64_t Q_fp16_addr = 0, K_fp16_cm_addr = 0, V_fp16_cm_addr = 0, P_fp16_addr = 0;
uint64_t P_sp_addr = 0, meta_P_addr = 0;
uint64_t Q_fp32_addr = 0, K_fp32_addr = 0, V_fp32_addr = 0, P_fp32_addr = 0;

// fp32 → fp16 (uint16 bits). Manual IEEE encode so it works on hosts
// without a native __fp16 type.
static inline uint16_t f2h_host(float x) {
  uint32_t bits;
  std::memcpy(&bits, &x, sizeof(bits));
  uint16_t sign = (bits >> 16) & 0x8000;
  uint16_t mant = (bits >> 13) & 0x03FF;
  int32_t  exp  = (int32_t)((bits >> 23) & 0xFF) - 127 + 15;
  if (exp >= 31)  return sign | 0x7C00;  // inf / overflow
  if (exp <= 0)   return sign;            // zero / denormal → flush to 0
  return sign | ((uint16_t)exp << 10) | mant;
}
static inline float h2f_host(uint16_t x) {
  uint32_t sign = (uint32_t)(x & 0x8000) << 16;
  uint32_t exp  = (x >> 10) & 0x1F;
  uint32_t mant = (x & 0x3FF);
  uint32_t bits;
  if (exp == 0) {
    bits = sign; // zero (flushed)
  } else if (exp == 31) {
    bits = sign | 0x7F800000 | (mant << 13); // inf / nan
  } else {
    bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

static void show_usage() {
  std::cout << "Vortex Attention Benchmark." << std::endl;
  std::cout << "Usage: [-n N] [-d D] [-t mode 0=SIMT 1=TCU 2=SPARSE] [-k kernel] [-h]" << std::endl;
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "n:d:t:k:h")) != -1) {
    switch (c) {
      case 'n': N_req = atoi(optarg); break;
      case 'd': d_req = atoi(optarg); break;
      case 't': mode  = atoi(optarg); break;
      case 'k': kernel_file = optarg; break;
      case 'h': show_usage(); exit(0);
      default:  show_usage(); exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    vx_mem_free(Q_buffer);
    vx_mem_free(K_buffer);
    vx_mem_free(S_buffer);
    vx_mem_free(P_buffer);
    vx_mem_free(V_buffer);
    vx_mem_free(O_buffer);
    if (Q_fp16_buffer)    vx_mem_free(Q_fp16_buffer);
    if (K_fp16_cm_buffer) vx_mem_free(K_fp16_cm_buffer);
    if (V_fp16_cm_buffer) vx_mem_free(V_fp16_cm_buffer);
    if (P_fp16_buffer)    vx_mem_free(P_fp16_buffer);
    if (P_sp_buffer)      vx_mem_free(P_sp_buffer);
    if (meta_P_buffer)    vx_mem_free(meta_P_buffer);
    if (Qsp_buffer)       vx_mem_free(Qsp_buffer);
    if (meta_Q_buffer)    vx_mem_free(meta_Q_buffer);
    vx_mem_free(krnl_buffer);
    vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

// Read kernel_cycles back from the device-side kernel_arg and print a tagged line.
static void read_back_cycles(const char* stage_tag) {
  kernel_arg_t back = {};
  vx_copy_from_dev(&back, args_buffer, 0, sizeof(kernel_arg_t));
  printf("KCYC[%s,mode=%u,nt=%u]: %lu\n",
         stage_tag, (unsigned)mode, (unsigned)NUM_THREADS,
         (unsigned long)back.kernel_cycles);
}

int main(int argc, char* argv[]) {
  parse_args(argc, argv);
  std::srand(50);

  std::cout << "open device connection" << std::endl;
  RT_CHECK(vx_dev_open(&device));

  uint64_t NT_caps = 0;
  vx_dev_caps(device, VX_CAPS_NUM_THREADS, &NT_caps);
  if ((uint32_t)NT_caps != NUM_THREADS) {
    printf("Error: device NUM_THREADS=%u but benchmark was built with NUM_THREADS=%u\n",
           (unsigned)NT_caps, (unsigned)NUM_THREADS);
    cleanup();
    return -1;
  }

  // Padding for TCU: N must be multiple of max(TM,TK) to feed both Q and P/V matmul.
  // d must be multiple of TK (Q's K-dim) AND TN (V's col-dim when fed into P@V).
  uint32_t N = N_req;
  uint32_t d = d_req;
  if (mode != MODE_SIMT) {
    // N_dim is used as (a) M dim of Q@K output rows [needs TM alignment],
    //                  (b) N dim of Q@K output cols [needs TN alignment],
    //                  (c) K dim of P@V [needs TK alignment].
    // d_dim is used as (a) K dim of Q@K [needs TK alignment],
    //                  (b) N dim of P@V output cols [needs TN alignment].
    uint32_t n_align = std::max({TM, TN, TK});
    uint32_t d_align = std::max(TK, TN);
    N = ((N_req + n_align - 1) / n_align) * n_align;
    d = ((d_req + d_align - 1) / d_align) * d_align;
    std::cout << "TCU mode — padded N=" << N << " (req " << N_req << "), "
              << "d=" << d << " (req " << d_req << ") "
              << "TM=" << TM << " TN=" << TN << " TK=" << TK << std::endl;
  } else {
    std::cout << "SIMT mode — N=" << N << " d=" << d << std::endl;
  }

  // Allocate padded host buffers (zero-filled outside the requested region so
  // the reference CPU matmul on the padded region yields the same values as on
  // the unpadded region for the valid submatrix).
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

  uint32_t in_size  = N * d;
  uint32_t s_size   = N * N;
  uint32_t out_size = N * d;
  uint32_t in_bytes  = in_size  * sizeof(float);
  uint32_t s_bytes   = s_size   * sizeof(float);
  uint32_t out_bytes = out_size * sizeof(float);

  // All buffers RW: we reuse them across 3 stages (stage 1 writes S, stage 2 reads S
  // and writes P, stage 3 reads P and writes O). Vortex doesn't strictly enforce these
  // flags but some allocators use them as placement hints, so being explicit avoids
  // surprises.
  RT_CHECK(vx_mem_alloc(device, in_bytes,  VX_MEM_READ_WRITE, &Q_buffer));
  RT_CHECK(vx_mem_address(Q_buffer, &Q_fp32_addr));
  RT_CHECK(vx_mem_alloc(device, in_bytes,  VX_MEM_READ_WRITE, &K_buffer));
  RT_CHECK(vx_mem_address(K_buffer, &K_fp32_addr));
  RT_CHECK(vx_mem_alloc(device, s_bytes,   VX_MEM_READ_WRITE, &S_buffer));
  RT_CHECK(vx_mem_address(S_buffer, &kernel_arg.S_addr));
  RT_CHECK(vx_mem_alloc(device, s_bytes,   VX_MEM_READ_WRITE, &P_buffer));
  RT_CHECK(vx_mem_address(P_buffer, &P_fp32_addr));
  RT_CHECK(vx_mem_alloc(device, in_bytes,  VX_MEM_READ_WRITE, &V_buffer));
  RT_CHECK(vx_mem_address(V_buffer, &V_fp32_addr));
  RT_CHECK(vx_mem_alloc(device, out_bytes, VX_MEM_READ_WRITE, &O_buffer));
  RT_CHECK(vx_mem_address(O_buffer, &kernel_arg.O_addr));

  kernel_arg.Q_addr = Q_fp32_addr;
  kernel_arg.K_addr = K_fp32_addr;
  kernel_arg.V_addr = V_fp32_addr;
  kernel_arg.P_addr = P_fp32_addr;
  kernel_arg.N = N;
  kernel_arg.d = d;

  // TCU fp16 buffers — sgemm_tcu storage contract
  if (mode != MODE_SIMT) {
    uint32_t in_fp16_bytes = in_size  * sizeof(uint16_t);
    uint32_t s_fp16_bytes  = s_size   * sizeof(uint16_t);
    RT_CHECK(vx_mem_alloc(device, in_fp16_bytes, VX_MEM_READ_WRITE, &Q_fp16_buffer));
    RT_CHECK(vx_mem_address(Q_fp16_buffer, &Q_fp16_addr));
    RT_CHECK(vx_mem_alloc(device, in_fp16_bytes, VX_MEM_READ_WRITE, &K_fp16_cm_buffer));
    RT_CHECK(vx_mem_address(K_fp16_cm_buffer, &K_fp16_cm_addr));
    RT_CHECK(vx_mem_alloc(device, in_fp16_bytes, VX_MEM_READ_WRITE, &V_fp16_cm_buffer));
    RT_CHECK(vx_mem_address(V_fp16_cm_buffer, &V_fp16_cm_addr));
    RT_CHECK(vx_mem_alloc(device, s_fp16_bytes,  VX_MEM_READ_WRITE, &P_fp16_buffer));
    RT_CHECK(vx_mem_address(P_fp16_buffer, &P_fp16_addr));
  }

  if (mode == MODE_SPARSE_TCU) {
    uint32_t psp_bytes = (N * N / 2) * sizeof(uint16_t);
    RT_CHECK(vx_mem_alloc(device, psp_bytes, VX_MEM_READ_WRITE, &P_sp_buffer));
    RT_CHECK(vx_mem_address(P_sp_buffer, &P_sp_addr));
    uint32_t meta_p_bytes = 64 * 1024;
    RT_CHECK(vx_mem_alloc(device, meta_p_bytes, VX_MEM_READ_WRITE, &meta_P_buffer));
    RT_CHECK(vx_mem_address(meta_P_buffer, &meta_P_addr));
  }

  // Upload fp32 Q/K/V (used by SIMT kernels and by the host-side CPU reference).
  RT_CHECK(vx_copy_to_dev(Q_buffer, h_Q.data(), 0, in_bytes));
  RT_CHECK(vx_copy_to_dev(K_buffer, h_K.data(), 0, in_bytes));
  RT_CHECK(vx_copy_to_dev(V_buffer, h_V.data(), 0, in_bytes));

  // Sparse mode: prune Q to 2:4, compress, pack metadata. Must run BEFORE the
  // dense fp16 packing so the fp16 Q buffer sees the pruned values.
  std::vector<uint16_t> h_Q_fp16, h_Q_compressed;
  std::vector<uint8_t>  sparse_masks;
  std::vector<uint32_t> h_meta_Q;
  if (mode == MODE_SPARSE_TCU) {
    h_Q_fp16.resize(N * d);
    for (uint32_t i = 0; i < N * d; ++i) h_Q_fp16[i] = f2h_host(h_Q[i]);
    if (!vt::prune_2to4_matrix<vt::fp16>(h_Q_fp16.data(), N, d)) {
      printf("Error: prune_2to4_matrix failed for Q\n");
      cleanup();
      return -1;
    }
    for (uint32_t i = 0; i < N * d; ++i) h_Q[i] = h2f_host(h_Q_fp16[i]);

    h_Q_compressed.resize(N * (d / 2));
    if (!vt::compress_2to4_matrix<vt::fp16>(h_Q_compressed.data(),
                                            h_Q_fp16.data(),
                                            sparse_masks, N, d)) {
      printf("Error: compress_2to4_matrix failed for Q\n");
      cleanup();
      return -1;
    }
    pack_metadata(h_meta_Q, sparse_masks, N, d);

    uint32_t qsp_bytes  = (uint32_t)h_Q_compressed.size() * sizeof(uint16_t);
    uint32_t meta_bytes = (uint32_t)h_meta_Q.size() * sizeof(uint32_t);
    RT_CHECK(vx_mem_alloc(device, qsp_bytes,  VX_MEM_READ_WRITE, &Qsp_buffer));
    RT_CHECK(vx_mem_alloc(device, meta_bytes, VX_MEM_READ_WRITE, &meta_Q_buffer));
    RT_CHECK(vx_mem_address(meta_Q_buffer, &kernel_arg.meta_Q_addr));
    RT_CHECK(vx_copy_to_dev(Qsp_buffer,    h_Q_compressed.data(), 0, qsp_bytes));
    RT_CHECK(vx_copy_to_dev(meta_Q_buffer, h_meta_Q.data(),        0, meta_bytes));
    RT_CHECK(vx_copy_to_dev(Q_buffer,      h_Q.data(),             0, in_bytes));  // pruned fp32 Q for CPU ref
    std::cout << "SPARSE: Q compressed " << qsp_bytes << "B, meta " << meta_bytes
              << "B, masks " << sparse_masks.size() << " groups" << std::endl;
  }

  // TCU mode data-prep: fp16 Q + fp16 K col-major + fp16 V col-major.
  if (mode != MODE_SIMT) {
    std::vector<uint16_t> h_Q_cm(N * d);
    std::vector<uint16_t> h_K_cm(N * d);
    std::vector<uint16_t> h_V_cm(N * d);
    for (uint32_t i = 0; i < N * d; ++i) h_Q_cm[i] = f2h_host(h_Q[i]);
    for (uint32_t n = 0; n < N; ++n)
      for (uint32_t k = 0; k < d; ++k)
        h_K_cm[n * d + k] = f2h_host(h_K[k * N + n]);
    for (uint32_t d_idx = 0; d_idx < d; ++d_idx)
      for (uint32_t n = 0; n < N; ++n)
        h_V_cm[d_idx * N + n] = f2h_host(h_V[n * d + d_idx]);

    uint32_t in_fp16_bytes = N * d * sizeof(uint16_t);
    RT_CHECK(vx_copy_to_dev(Q_fp16_buffer,    h_Q_cm.data(), 0, in_fp16_bytes));
    RT_CHECK(vx_copy_to_dev(K_fp16_cm_buffer, h_K_cm.data(), 0, in_fp16_bytes));
    RT_CHECK(vx_copy_to_dev(V_fp16_cm_buffer, h_V_cm.data(), 0, in_fp16_bytes));
  }

  // upload kernel binary
  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));

  int errors = 0;
  const float atol = (mode == MODE_SIMT) ? ((d >= 128) ? 1e-3f : 1e-5f) : 2e-2f;
  const float rtol = (mode == MODE_SIMT) ? ((d >= 128) ? 1e-3f : 1e-5f) : 2e-2f;

  // ===================================================================
  // Fused flash-online-softmax SIMT path — single kernel launch.
  // Runs when mode == MODE_SIMT AND the head_dim is within the fused
  // template's supported set (HEAD_DIM ∈ {1,2,4,8,16}). Larger d falls
  // through to the unfused 3-stage SIMT path below.
  // This path uses K stored as [N x d] (per-token row layout).
  // ===================================================================
  const bool fused_simt_supported =
      (d_req == 1 || d_req == 2 || d_req == 4 || d_req == 8 || d_req == 16);
  if (mode == MODE_SIMT && fused_simt_supported) {
    // The fused kernel's templated flash_kernel_body<HEAD_DIM,BLOCK_SIZE_C>
    // supports HEAD_DIM ∈ {1,2,4,8,16} and BLOCK_SIZE_C ∈ {8,16,32,64,128}.
    // Pick shapes that match the requested N,d and fit in that template set.
    uint32_t head_dim = d_req;
    uint32_t block_size_r = std::min((uint32_t)8, N_req);
    uint32_t block_size_c = std::max(block_size_r, (uint32_t)8);

    // Rebuild K in [N × d] layout for the fused path. The original h_K
    // (above) was built in [d × N] layout for attention's matmul.
    std::vector<float> h_K_flash(N_req * d_req, 0.0f);
    for (uint32_t r = 0; r < d_req; ++r)
      for (uint32_t c = 0; c < N_req; ++c)
        h_K_flash[c * d_req + r] = h_K[r * N + c];
    // Similarly V is already [N × d] so just tighten to requested region.
    std::vector<float> h_V_flash(N_req * d_req, 0.0f);
    for (uint32_t r = 0; r < N_req; ++r)
      for (uint32_t c = 0; c < d_req; ++c)
        h_V_flash[r * d_req + c] = h_V[r * d + c];
    // Q in [N × d] — reuse from the padded buffer, but tighten.
    std::vector<float> h_Q_flash(N_req * d_req, 0.0f);
    for (uint32_t r = 0; r < N_req; ++r)
      for (uint32_t c = 0; c < d_req; ++c)
        h_Q_flash[r * d_req + c] = h_Q[r * d + c];
    std::vector<float> h_O_flash(N_req * d_req, 0.0f);

    // Separate device buffers for the fused path (Q, K, V, O in [N×d] layout).
    uint32_t nd_bytes = N_req * d_req * sizeof(float);
    vx_buffer_h Qf = nullptr, Kf = nullptr, Vf = nullptr, Of = nullptr;
    RT_CHECK(vx_mem_alloc(device, nd_bytes, VX_MEM_READ_WRITE, &Qf));
    RT_CHECK(vx_mem_alloc(device, nd_bytes, VX_MEM_READ_WRITE, &Kf));
    RT_CHECK(vx_mem_alloc(device, nd_bytes, VX_MEM_READ_WRITE, &Vf));
    RT_CHECK(vx_mem_alloc(device, nd_bytes, VX_MEM_READ_WRITE, &Of));
    RT_CHECK(vx_mem_address(Qf, &kernel_arg.Q_addr));
    RT_CHECK(vx_mem_address(Kf, &kernel_arg.K_addr));
    RT_CHECK(vx_mem_address(Vf, &kernel_arg.V_addr));
    RT_CHECK(vx_mem_address(Of, &kernel_arg.O_addr));
    RT_CHECK(vx_copy_to_dev(Qf, h_Q_flash.data(), 0, nd_bytes));
    RT_CHECK(vx_copy_to_dev(Kf, h_K_flash.data(), 0, nd_bytes));
    RT_CHECK(vx_copy_to_dev(Vf, h_V_flash.data(), 0, nd_bytes));

    kernel_arg.seq_len       = N_req;
    kernel_arg.head_dim      = head_dim;
    kernel_arg.block_size_r  = block_size_r;
    kernel_arg.block_size_c  = block_size_c;
    kernel_arg.kernel_type   = 0;              // SIMT
    kernel_arg.kernel_id     = KID_FLASH_FUSED;
    kernel_arg.grid_dim[0]   = N_req / block_size_r;
    kernel_arg.grid_dim[1]   = 1;
    kernel_arg.block_dim[0]  = block_size_r;
    kernel_arg.block_dim[1]  = 1;

    // Upload kernel binary if not done yet
    if (krnl_buffer == nullptr) {
      RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
    }
    if (args_buffer == nullptr) {
      RT_CHECK(vx_mem_alloc(device, sizeof(kernel_arg_t), VX_MEM_READ_WRITE, &args_buffer));
    }
    RT_CHECK(vx_copy_to_dev(args_buffer, &kernel_arg, 0, sizeof(kernel_arg_t)));
    RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
    RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

    // Read cycles
    {
      kernel_arg_t back = {};
      vx_copy_from_dev(&back, args_buffer, 0, sizeof(kernel_arg_t));
      printf("KCYC[QK,mode=0,nt=%u]: %lu\n", (unsigned)NUM_THREADS, (unsigned long)back.kernel_cycles);
      printf("KCYC[SM,mode=0,nt=%u]: 0\n", (unsigned)NUM_THREADS);
      printf("KCYC[PV,mode=0,nt=%u]: 0\n", (unsigned)NUM_THREADS);
    }

    RT_CHECK(vx_copy_from_dev(h_O_flash.data(), Of, 0, nd_bytes));

    std::vector<float> h_O_ref(N_req * d_req, 0.0f);
    attention_cpu_fused(h_O_ref.data(), h_Q_flash.data(), h_K_flash.data(),
                         h_V_flash.data(), N_req, d_req);
    const float simt_atol = 1e-4f;
    const float simt_rtol = 1e-4f;
    for (uint32_t r = 0; r < N_req; ++r)
      for (uint32_t c = 0; c < d_req; ++c) {
        int idx = r * d_req + c;
        if (!compare(h_O_ref[idx], h_O_flash[idx], simt_atol, simt_rtol, idx, errors, "Oflash"))
          ++errors;
      }

    if (Qf) vx_mem_free(Qf);
    if (Kf) vx_mem_free(Kf);
    if (Vf) vx_mem_free(Vf);
    if (Of) vx_mem_free(Of);

    cleanup();
    if (errors != 0) {
      std::cout << "Found " << errors << " errors in fused flash SIMT path!" << std::endl;
      std::cout << "FAILED!" << std::endl;
      return errors;
    }
    std::cout << "PASSED!" << std::endl;
    return 0;
  }

  // -------------------------------------------------------------------
  // Stage 1: S = Q @ K^T
  // -------------------------------------------------------------------
  std::cout << "=== Stage 1: S = Q @ K^T ===" << std::endl;
  if (mode == MODE_SIMT) {
    kernel_arg.Q_addr = Q_fp32_addr;
    kernel_arg.K_addr = K_fp32_addr;
    kernel_arg.grid_dim[0] = N;
    kernel_arg.grid_dim[1] = N;
    kernel_arg.block_dim[0] = 0;
    kernel_arg.block_dim[1] = 0;
    kernel_arg.kernel_id = KID_QK_SIMT;
  } else if (mode == MODE_DENSE_TCU) {
    kernel_arg.Q_addr = Q_fp16_addr;
    kernel_arg.K_addr = K_fp16_cm_addr;
    kernel_arg.grid_dim[0]  = N / TN;
    kernel_arg.grid_dim[1]  = N / TM;
    kernel_arg.block_dim[0] = NUM_THREADS;
    kernel_arg.block_dim[1] = 1;
    kernel_arg.kernel_id    = KID_QK_TCU;
  } else { // MODE_SPARSE_TCU
    uint64_t qsp_addr = 0;
    RT_CHECK(vx_mem_address(Qsp_buffer, &qsp_addr));
    kernel_arg.Q_addr = qsp_addr;
    kernel_arg.K_addr = K_fp16_cm_addr;
    kernel_arg.grid_dim[0]  = N / TN;
    kernel_arg.grid_dim[1]  = N / TM;
    kernel_arg.block_dim[0] = NUM_THREADS;
    kernel_arg.block_dim[1] = 1;
    kernel_arg.kernel_id    = KID_QK_SPARSE;
  }
  // args_buffer must be RW because the kernel writes kernel_cycles back to it.
  // vx_upload_bytes forces VX_MEM_READ, so allocate + copy manually the first time,
  // and only vx_copy_to_dev on subsequent stages.
  if (args_buffer == nullptr) {
    RT_CHECK(vx_mem_alloc(device, sizeof(kernel_arg_t), VX_MEM_READ_WRITE, &args_buffer));
  }
  RT_CHECK(vx_copy_to_dev(args_buffer, &kernel_arg, 0, sizeof(kernel_arg_t)));
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
  read_back_cycles("QK");

  std::vector<float> h_S(s_size);
  RT_CHECK(vx_copy_from_dev(h_S.data(), S_buffer, 0, s_bytes));

  // Verify only the valid (N_req × N_req) region. Padded rows/cols are undefined
  // on the device because padded Q rows are zero so S[pad, *] = 0; softmax of all
  // zeros would explode, so we zero the padded region explicitly post-hoc before
  // softmax.
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
    errors = 0; // keep going so we measure all three kernels
  }

  // Zero pad rows/cols in S before softmax so the padded rows don't blow up
  // the running max in stage 2. Softmax kernel is SIMT and unaware of padding.
  if (mode != MODE_SIMT) {
    for (uint32_t r = 0; r < N; ++r) {
      for (uint32_t c = 0; c < N; ++c) {
        if (r >= N_req || c >= N_req) h_S[r * N + c] = -1e9f;  // softmax -> ~0
      }
    }
    RT_CHECK(vx_copy_to_dev(S_buffer, h_S.data(), 0, s_bytes));
  }

  // -------------------------------------------------------------------
  // Stage 2: P = softmax(S)  — always SIMT
  // -------------------------------------------------------------------
  std::cout << "=== Stage 2: P = softmax(S) ===" << std::endl;
  kernel_arg.grid_dim[0] = N;
  kernel_arg.grid_dim[1] = 1;
  kernel_arg.block_dim[0] = 0;
  kernel_arg.block_dim[1] = 0;
  kernel_arg.kernel_id = KID_SOFTMAX;
  // args_buffer must be RW because the kernel writes kernel_cycles back to it.
  // vx_upload_bytes forces VX_MEM_READ, so allocate + copy manually the first time,
  // and only vx_copy_to_dev on subsequent stages.
  if (args_buffer == nullptr) {
    RT_CHECK(vx_mem_alloc(device, sizeof(kernel_arg_t), VX_MEM_READ_WRITE, &args_buffer));
  }
  RT_CHECK(vx_copy_to_dev(args_buffer, &kernel_arg, 0, sizeof(kernel_arg_t)));
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
  read_back_cycles("SM");

  std::vector<float> h_P(s_size);
  RT_CHECK(vx_copy_from_dev(h_P.data(), P_buffer, 0, s_bytes));
  {
    std::vector<float> h_P_ref(N * N);
    softmax_cpu(h_P_ref.data(), h_S.data(), N, N);
    for (uint32_t r = 0; r < N_req; ++r)
      for (uint32_t c = 0; c < N_req; ++c) {
        int idx = r * N + c;
        // softmax is fp32 on both sides, tight tolerance is fine
        if (!compare(h_P_ref[idx], h_P[idx], 1e-5f, 1e-5f, idx, errors, "P"))
          ++errors;
      }
  }
  if (errors) {
    std::cout << "P stage found " << errors << " errors, continuing" << std::endl;
    errors = 0;
  }

  // -------------------------------------------------------------------
  // Stage 3: O = P @ V
  // -------------------------------------------------------------------
  std::cout << "=== Stage 3: O = P @ V ===" << std::endl;

  if (mode == MODE_DENSE_TCU) {
    std::vector<uint16_t> h_P_fp16(N * N);
    for (uint32_t i = 0; i < N * N; ++i) h_P_fp16[i] = f2h_host(h_P[i]);
    RT_CHECK(vx_copy_to_dev(P_fp16_buffer, h_P_fp16.data(), 0,
                            (uint32_t)h_P_fp16.size() * sizeof(uint16_t)));
  } else if (mode == MODE_SPARSE_TCU) {
    std::vector<uint16_t> h_P_fp16(N * N);
    for (uint32_t i = 0; i < N * N; ++i) h_P_fp16[i] = f2h_host(h_P[i]);
    if (!vt::prune_2to4_matrix<vt::fp16>(h_P_fp16.data(), N, N)) {
      printf("Error: prune_2to4_matrix failed for P\n");
      cleanup();
      return -1;
    }
    for (uint32_t i = 0; i < N * N; ++i) h_P[i] = h2f_host(h_P_fp16[i]);

    std::vector<uint16_t> h_P_compressed(N * (N / 2));
    std::vector<uint8_t>  sparse_masks_P;
    if (!vt::compress_2to4_matrix<vt::fp16>(h_P_compressed.data(),
                                            h_P_fp16.data(),
                                            sparse_masks_P, N, N)) {
      printf("Error: compress_2to4_matrix failed for P\n");
      cleanup();
      return -1;
    }
    std::vector<uint32_t> h_meta_P;
    pack_metadata(h_meta_P, sparse_masks_P, N, N);

    uint32_t psp_bytes   = (uint32_t)h_P_compressed.size() * sizeof(uint16_t);
    uint32_t pmeta_bytes = (uint32_t)h_meta_P.size()       * sizeof(uint32_t);
    RT_CHECK(vx_copy_to_dev(P_sp_buffer,   h_P_compressed.data(), 0, psp_bytes));
    RT_CHECK(vx_copy_to_dev(meta_P_buffer, h_meta_P.data(),       0, pmeta_bytes));
  }

  if (mode == MODE_SIMT) {
    kernel_arg.P_addr = P_fp32_addr;
    kernel_arg.V_addr = V_fp32_addr;
    kernel_arg.grid_dim[0] = d;
    kernel_arg.grid_dim[1] = N;
    kernel_arg.block_dim[0] = 0;
    kernel_arg.block_dim[1] = 0;
    kernel_arg.kernel_id = KID_PV_SIMT;
  } else if (mode == MODE_DENSE_TCU) {
    kernel_arg.P_addr = P_fp16_addr;
    kernel_arg.V_addr = V_fp16_cm_addr;
    kernel_arg.grid_dim[0]  = d / TN;
    kernel_arg.grid_dim[1]  = N / TM;
    kernel_arg.block_dim[0] = NUM_THREADS;
    kernel_arg.block_dim[1] = 1;
    kernel_arg.kernel_id    = KID_PV_TCU;
  } else { // MODE_SPARSE_TCU
    kernel_arg.P_addr      = P_sp_addr;
    kernel_arg.meta_P_addr = meta_P_addr;
    kernel_arg.V_addr      = V_fp16_cm_addr;
    kernel_arg.grid_dim[0]  = d / TN;
    kernel_arg.grid_dim[1]  = N / TM;
    kernel_arg.block_dim[0] = NUM_THREADS;
    kernel_arg.block_dim[1] = 1;
    kernel_arg.kernel_id    = KID_PV_SPARSE;
  }
  // args_buffer must be RW because the kernel writes kernel_cycles back to it.
  // vx_upload_bytes forces VX_MEM_READ, so allocate + copy manually the first time,
  // and only vx_copy_to_dev on subsequent stages.
  if (args_buffer == nullptr) {
    RT_CHECK(vx_mem_alloc(device, sizeof(kernel_arg_t), VX_MEM_READ_WRITE, &args_buffer));
  }
  RT_CHECK(vx_copy_to_dev(args_buffer, &kernel_arg, 0, sizeof(kernel_arg_t)));
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
  read_back_cycles("PV");

  std::vector<float> h_O(out_size);
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
