#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include <vx_tensor.h>
#include <vx_print.h>
#include <cmath>
#include <cstring>
#include "common.h"

#ifndef NUM_TCU_LANES
#define NUM_TCU_LANES NUM_THREADS
#endif

// TCU tile shape sized to the actual warp width.
namespace vt = vortex::tensor;
using tcu_ctx = vt::wmma_context<NUM_TCU_LANES, vt::fp16, vt::fp32>;
using sp_ctx  = vt::wmma_context<NUM_TCU_LANES, vt::fp16, vt::fp32, true>;
static constexpr uint32_t TCU_K = tcu_ctx::tileK;
static constexpr uint32_t TM = tcu_ctx::tileM;
static constexpr uint32_t TN = tcu_ctx::tileN;
static constexpr uint32_t TK = tcu_ctx::tileK;

static inline uint16_t f2h(float x) {
  __fp16 h = (__fp16)x;
  uint16_t out;
  memcpy(&out, &h, sizeof(out));
  return out;
}

static inline float h2f(uint16_t x) {
  __fp16 h;
  memcpy(&h, &x, sizeof(h));
  return (float)h;
}

// SIMT Flash Implementation
template<uint32_t HEAD_DIM, uint32_t BLOCK_SIZE_C>
void flash_kernel_body(kernel_arg_t *arg) {
  // Setup buffer arguments
  float* Q_ptr = reinterpret_cast<float*>(arg->Q_addr);
  float* K_ptr = reinterpret_cast<float*>(arg->K_addr);
  float* V_ptr = reinterpret_cast<float*>(arg->V_addr);
  float* O_ptr = reinterpret_cast<float*>(arg->O_addr);

  auto seq_len = arg->seq_len;
  auto block_size_r = arg->block_size_r;

  // Allocate local memory
  auto local_ptr = __local_mem((block_size_r + 2 * BLOCK_SIZE_C) * HEAD_DIM * sizeof(float));
  auto local_Q = (float*)local_ptr;
  auto local_K = (float*)local_Q + block_size_r * HEAD_DIM;
  auto local_V = (float*)local_K + BLOCK_SIZE_C * HEAD_DIM;

  // Determine global/local row index
  auto g_row = blockIdx.x * blockDim.x + threadIdx.x;
  auto l_row = threadIdx.x;
  auto g_row_offset = g_row * HEAD_DIM;
  auto l_row_offset = l_row * HEAD_DIM;

  // Load Q_i from HBM
  for (uint32_t k = 0; k < HEAD_DIM; ++k)
    local_Q[l_row_offset + k] = Q_ptr[g_row_offset + k];

  // Thread's row of Q block
  float* Q_row = local_Q + l_row * HEAD_DIM;

  // Initialize O_i
  float O_buf[HEAD_DIM];
  for (uint32_t k = 0; k < HEAD_DIM; ++k)
    O_buf[k] = 0.0f;

  // Create buffer to store row of S and P
  float sp_buf[BLOCK_SIZE_C];

  // Initialize m_i (rowmax), l_i (softmax denominator)
  float m = -INFINITY;
  float l = 0.0f;

  // Loop over blocks of K and V
  for (uint32_t j = 0; j < seq_len; j += BLOCK_SIZE_C) {
    auto block_offset = j * HEAD_DIM;

    // Load K_j and V_j^T
    for (uint32_t k = 0; k < BLOCK_SIZE_C / block_size_r; ++k) {
      auto row = k * block_size_r + l_row;
      auto row_offset = row * HEAD_DIM;
      for (uint32_t col = 0; col < HEAD_DIM; ++col) {
        auto offset = row_offset + col;
        local_K[offset] = K_ptr[block_offset + offset];
        local_V[col * BLOCK_SIZE_C + row] = V_ptr[block_offset + offset];
      }
    }

    __syncthreads();

    // Compute S_ij, the dot product of thread's Q row and each row of K_j
    for (uint32_t k = 0; k < BLOCK_SIZE_C; ++k)
      sp_buf[k] = 0;
    for (uint32_t k = 0; k < BLOCK_SIZE_C; ++k) 
      for (uint32_t elem = 0; elem < HEAD_DIM; ++elem)
        sp_buf[k] += Q_row[elem] * local_K[k * HEAD_DIM + elem];

    // Row max
    float rowmax = sp_buf[0];
    for (uint32_t k = 1; k < BLOCK_SIZE_C; ++k)
      rowmax = (sp_buf[k] > rowmax ? sp_buf[k] : rowmax);

    // Compute P_ij, the softmax numerators of S_ij
    for (uint32_t k = 0; k < BLOCK_SIZE_C; ++k)
      sp_buf[k] = expf(sp_buf[k] - rowmax);

    // Row sum
    float rowsum = 0.0f;
    for (uint32_t k = 0; k < BLOCK_SIZE_C; ++k)
      rowsum += sp_buf[k];

    // Compute new m and l
    float new_m = (m > rowmax ? m : rowmax);
    float new_l = expf(m - new_m) * l + expf(rowmax - new_m) * rowsum;

    // Weights of old and new O
    float old_weight = expf(m - new_m);
    float new_weight = expf(rowmax - new_m);

    // Update O
    for (uint32_t k = 0; k < HEAD_DIM; ++k) {
      // Compute dot product of thread's P_ij and each col of V_j
      float dot = 0.0f;
      for (uint32_t elem = 0; elem < BLOCK_SIZE_C; ++elem)
        dot += sp_buf[elem] * local_V[k * BLOCK_SIZE_C + elem]; 
      O_buf[k] = old_weight * O_buf[k] + new_weight * dot;
    }

    // Update m and l for next block
    m = new_m;
    l = new_l;

    __syncthreads();
  }

  // Normalize O by softmax denominator and write back to HBM
  float inv_l = 1.0f / l;
  for (uint32_t k = 0; k < HEAD_DIM; ++k)
    O_ptr[g_row_offset + k] = O_buf[k] * inv_l;
}

// TCU Flash Implementation
static void flashattention_tcu(kernel_arg_t* arg) {
  if (arg->head_dim != 8 || arg->block_size_r != 8 || arg->block_size_c != 8) {
    return;
  }

  float* Q_ptr = reinterpret_cast<float*>(arg->Q_addr);
  float* K_ptr = reinterpret_cast<float*>(arg->K_addr);
  float* V_ptr = reinterpret_cast<float*>(arg->V_addr);
  float* O_ptr = reinterpret_cast<float*>(arg->O_addr);

  auto seq_len = arg->seq_len;

  auto local_ptr = __local_mem(
    3 * TCU_K * TCU_K * sizeof(uint16_t) +  // Q, K, Vh
    TCU_K * TCU_K * sizeof(float) +          // S  
    8 * sizeof(float) +                      // row_m
    8 * sizeof(float) +                      // row_l
    8 * 8 * sizeof(float) +                  // row_O
    8 * sizeof(float) +                      // shared_maxval
    8 * sizeof(float)                        // shared_sumval
  );
  
  uint16_t* local_Q  = reinterpret_cast<uint16_t*>(local_ptr);
  uint16_t* local_K  = local_Q  + TCU_K * TCU_K;
  uint16_t* local_Vh = local_K  + TCU_K * TCU_K;
  float*    local_S  = reinterpret_cast<float*>(local_Vh + TCU_K * TCU_K);
  
  float* row_m = local_S + TCU_K * TCU_K;
  float* row_l = row_m + 8;
  float* row_O = row_l + 8;
  float* shared_maxval = row_O + 64;
  float* shared_sumval = shared_maxval + 8;

  uint32_t tile_row = blockIdx.x;
  uint32_t tid = threadIdx.x;
  uint32_t global_row_start = tile_row * 8;
  
  // Load Q tile (zero first, then load 8x8 data)
  for (uint32_t i = tid; i < TCU_K * TCU_K; i += blockDim.x) {
    local_Q[i] = 0;
  }
  for (uint32_t local_r = 0; local_r < 8; local_r++) {
    for (uint32_t c = tid; c < 8; c += blockDim.x) {
      uint32_t global_row = global_row_start + local_r;
      local_Q[local_r * TCU_K + c] = f2h(Q_ptr[global_row * 8 + c]);
    }
  }

  // Initialize stats
  if (tid < 8) {
    row_m[tid] = -INFINITY;
    row_l[tid] = 0.0f;
    for (uint32_t c = 0; c < 8; ++c) {
        row_O[tid * 8 + c] = 0.0f;
    }
  }

  tcu_ctx::fragment_a   fragA;
  tcu_ctx::fragment_b   fragB;
  tcu_ctx::fragment_acc fragC;
  tcu_ctx::fragment_b   fragV;
  tcu_ctx::fragment_acc fragPV;

  __syncthreads();

  for (uint32_t kv_block = 0; kv_block < seq_len / 8; kv_block++) {
    uint32_t kv_row_start = kv_block * 8;
    
    // Load K row-major (col_major load will transpose for Q*K^T)
    for (uint32_t i = tid; i < TCU_K * TCU_K; i += blockDim.x) {
      local_K[i] = 0;
    }
    for (uint32_t local_r = 0; local_r < 8; local_r++) {
      for (uint32_t c = tid; c < 8; c += blockDim.x) {
        uint32_t global_row = kv_row_start + local_r;
        local_K[local_r * TCU_K + c] = f2h(K_ptr[global_row * 8 + c]);
      }
    }

    // Load V (transposed for PV = P*V; col_major load gives B=V)
    for (uint32_t i = tid; i < TCU_K * TCU_K; i += blockDim.x) {
      local_Vh[i] = 0;
    }
    for (uint32_t local_r = 0; local_r < 8; local_r++) {
      for (uint32_t c = tid; c < 8; c += blockDim.x) {
        uint32_t global_row = kv_row_start + local_r;
        local_Vh[c * TCU_K + local_r] = f2h(V_ptr[global_row * 8 + c]);
      }
    }

    __syncthreads();

    // S = Q * K^T
    tcu_ctx::fill_fragment(fragC, 0);
    tcu_ctx::load_matrix_sync(fragA, local_Q, TCU_K);
    tcu_ctx::load_matrix_sync<vt::col_major>(fragB, local_K, TCU_K);
    tcu_ctx::mma_sync(fragC, fragA, fragB, fragC);
    tcu_ctx::store_matrix_sync(local_S, fragC, TCU_K);  // Stores with stride TCU_K!

    __syncthreads();

    // SOFTMAX
    if (tid < 8) {
      uint32_t r = tid;
      float rowmax = -INFINITY;
      for (uint32_t c = 0; c < 8; ++c) {
        // CRITICAL: Use TCU_K as stride when reading from local_S!
        float val = local_S[r * TCU_K + c];
        if (val > rowmax) rowmax = val;
      }
      shared_maxval[r] = rowmax;
    }
    __syncthreads();

    // Compute exp (all threads)
    for (uint32_t local_r = 0; local_r < 8; local_r++) {
      for (uint32_t c = tid; c < 8; c += blockDim.x) {
        // Use TCU_K stride for both read and write
        float s_val = local_S[local_r * TCU_K + c];
        float prob = expf(s_val - shared_maxval[local_r]);
        local_K[local_r * TCU_K + c] = f2h(prob);
        local_S[local_r * TCU_K + c] = prob;  // Reuse local_S for prob
      }
    }
    __syncthreads();

    // Compute sum
    if (tid < 8) {
      uint32_t r = tid;
      float rowsum = 0.0f;
      for (uint32_t c = 0; c < 8; ++c) {
        rowsum += local_S[r * TCU_K + c];  // TCU_K stride
      }
      shared_sumval[r] = rowsum;
    }
    __syncthreads();

    // Zero P buffer, then update stats and write weighted P
    for (uint32_t i = tid; i < TCU_K * TCU_K; i += blockDim.x) {
      local_K[i] = 0;
    }
    __syncthreads();

    if (tid < 8) {
      uint32_t r = tid;
      float rowmax = shared_maxval[r];
      float rowsum = shared_sumval[r];

      float m_old = row_m[r];
      float l_old = row_l[r];
      float m_new = fmaxf(m_old, rowmax);
      float l_new = expf(m_old - m_new) * l_old + expf(rowmax - m_new) * rowsum;

      float w_old = expf(m_old - m_new);
      float w_new = expf(rowmax - m_new);

      // Scale existing output
      for (uint32_t c = 0; c < 8; ++c) {
        row_O[r * 8 + c] *= w_old;
      }

      // Weight probabilities into zeroed P buffer
      for (uint32_t c = 0; c < 8; ++c) {
        float p = local_S[r * TCU_K + c];  // read prob from local_S
        local_K[r * TCU_K + c] = f2h(w_new * p);
      }

      row_m[r] = m_new;
      row_l[r] = l_new;
    }
    __syncthreads();

    // PV = P * V
    tcu_ctx::fill_fragment(fragPV, 0);
    tcu_ctx::load_matrix_sync(fragA, local_K, TCU_K);
    tcu_ctx::load_matrix_sync<vt::col_major>(fragV, local_Vh, TCU_K);
    tcu_ctx::mma_sync(fragPV, fragA, fragV, fragPV);
    tcu_ctx::store_matrix_sync(local_S, fragPV, TCU_K);  // Stores with stride TCU_K!
    asm volatile ("fence" ::: "memory");

    __syncthreads();

    // Accumulate PV into O
    if (tid < 8) {
      uint32_t r = tid;
      for (uint32_t c = 0; c < 8; ++c) {
        // CRITICAL: Read from local_S with TCU_K stride!
        row_O[r * 8 + c] += local_S[r * TCU_K + c];
      }
    }

    __syncthreads();
  }

  // Final normalization and write
  if (tid < 8) {
    uint32_t local_r = tid;
    uint32_t global_row = global_row_start + local_r;
    float inv_l = 1.0f / row_l[local_r];
    
    for (uint32_t c = 0; c < 8; ++c) {
      O_ptr[global_row * 8 + c] = row_O[local_r * 8 + c] * inv_l;
    }
  }
}

// =============================================================================
// SIMT unfused attention kernels — one thread per output cell.
// =============================================================================

void flash_qk_simt(kernel_arg_t* __UNIFORM__ arg) {
  auto Q = reinterpret_cast<float*>(arg->Q_addr);
  auto K = reinterpret_cast<float*>(arg->K_addr);
  auto S = reinterpret_cast<float*>(arg->S_addr);
  uint32_t N = arg->N;
  uint32_t d = arg->d;
  int col = blockIdx.x;
  int row = blockIdx.y;
  if (row < (int)N && col < (int)N) {
    float sum = 0;
    for (uint32_t e = 0; e < d; ++e) sum += Q[row * d + e] * K[e * N + col];
    S[row * N + col] = sum;
  }
}

void flash_pv_simt(kernel_arg_t* __UNIFORM__ arg) {
  auto P = reinterpret_cast<float*>(arg->P_addr);
  auto V = reinterpret_cast<float*>(arg->V_addr);
  auto O = reinterpret_cast<float*>(arg->O_addr);
  uint32_t N = arg->N;
  uint32_t d = arg->d;
  int col = blockIdx.x;
  int row = blockIdx.y;
  if (row < (int)N && col < (int)d) {
    float sum = 0;
    for (uint32_t e = 0; e < N; ++e) sum += P[row * N + e] * V[e * d + col];
    O[row * d + col] = sum;
  }
}

// =============================================================================
// Unfused attention-style TCU kernels (dense + sparse) — used when the fused
// flashattention_tcu path's hardcoded 8x8 tiles don't line up with this NT.
// Single-warp: iterate all tile_rows inside.
// =============================================================================

// Dense Q@Kᵀ → S.  Mirrors sgemm_tcu/kernel.cpp verbatim. One output tile
// per block via (blockIdx.y, blockIdx.x). Direct global→fragment loads.
// Host pre-packs Q as fp16 row-major [N×d] and K as fp16 col-major [N×d].
void flash_qk_tcu(kernel_arg_t* __UNIFORM__ arg) {
  auto pA = reinterpret_cast<tcu_ctx::input_t*>(arg->Q_addr);
  auto pB = reinterpret_cast<tcu_ctx::input_t*>(arg->K_addr);
  auto pC = reinterpret_cast<tcu_ctx::output_t*>(arg->S_addr);
  uint32_t N_attn = arg->N;
  uint32_t d_attn = arg->d;

  tcu_ctx::fragment_a   fragA;
  tcu_ctx::fragment_b   fragB;
  tcu_ctx::fragment_acc fragC;

  uint32_t tile_row = blockIdx.y * tcu_ctx::tileM;
  uint32_t tile_col = blockIdx.x * tcu_ctx::tileN;

  tcu_ctx::fill_fragment(fragC, 0);

  for (int i = 0; i < (int)d_attn; i += (int)tcu_ctx::tileK) {
    auto pTileA = pA + tile_row * d_attn + i;
    auto pTileB = pB + tile_col * d_attn + i;
    tcu_ctx::load_matrix_sync(fragA, pTileA, d_attn);
    tcu_ctx::load_matrix_sync<vt::col_major>(fragB, pTileB, d_attn);
    tcu_ctx::mma_sync(fragC, fragA, fragB, fragC);
  }

  auto pTileC = pC + tile_row * N_attn + tile_col;
  tcu_ctx::store_matrix_sync(pTileC, fragC, N_attn);
}

// Dense P@V → O.  Same sgemm_tcu pattern as flash_qk_tcu, but the GEMM is
// M=N_attn, N=d_attn, K=N_attn with A=P[N×N] and B=V[d×N] col-major.
void flash_pv_tcu(kernel_arg_t* __UNIFORM__ arg) {
  auto pA = reinterpret_cast<tcu_ctx::input_t*>(arg->P_addr);
  auto pB = reinterpret_cast<tcu_ctx::input_t*>(arg->V_addr);
  auto pC = reinterpret_cast<tcu_ctx::output_t*>(arg->O_addr);
  uint32_t N_attn = arg->N;
  uint32_t d_attn = arg->d;

  tcu_ctx::fragment_a   fragA;
  tcu_ctx::fragment_b   fragB;
  tcu_ctx::fragment_acc fragC;

  uint32_t tile_row = blockIdx.y * tcu_ctx::tileM;
  uint32_t tile_col = blockIdx.x * tcu_ctx::tileN;

  tcu_ctx::fill_fragment(fragC, 0);

  for (int i = 0; i < (int)N_attn; i += (int)tcu_ctx::tileK) {
    auto pTileA = pA + tile_row * N_attn + i;
    auto pTileB = pB + tile_col * N_attn + i;
    tcu_ctx::load_matrix_sync(fragA, pTileA, N_attn);
    tcu_ctx::load_matrix_sync<vt::col_major>(fragB, pTileB, N_attn);
    tcu_ctx::mma_sync(fragC, fragA, fragB, fragC);
  }

  auto pTileC = pC + tile_row * d_attn + tile_col;
  tcu_ctx::store_matrix_sync(pTileC, fragC, d_attn);
}

// Sparse Q@Kᵀ with compressed fp16 Q + metadata. Mirrors sgemm_tcu_sp.
void flash_qk_sparse(kernel_arg_t* __UNIFORM__ arg) {
  auto pA = reinterpret_cast<sp_ctx::input_t*>(arg->Q_addr);       // compressed fp16 Q
  auto pB = reinterpret_cast<sp_ctx::input_t*>(arg->K_addr);       // fp16 K (col-major)
  auto pC = reinterpret_cast<sp_ctx::output_t*>(arg->S_addr);
  auto pMetaSpBase = reinterpret_cast<const float*>(arg->meta_Q_addr);

  uint32_t N_attn = arg->N;
  uint32_t d_attn = arg->d;
  uint32_t stride_A = d_attn / 2;

  sp_ctx::fragment_a   fragA;
  sp_ctx::fragment_b   fragB;
  sp_ctx::fragment_acc fragC;

  uint32_t tile_row = blockIdx.y * sp_ctx::tileM;
  uint32_t tile_col = blockIdx.x * sp_ctx::tileN;

  sp_ctx::fill_fragment(fragC, 0);

  constexpr uint32_t rtl_i_ratio = 32 / vt::fp16::bits;
  constexpr uint32_t meta_cols = (NUM_TCU_LANES * 2 * rtl_i_ratio + 31) / 32;
  using kcfg = vt::wmma_config_t<NUM_TCU_LANES>;
  constexpr uint32_t PD = kcfg::m_steps * (kcfg::k_steps / 2);
  constexpr uint32_t num_meta_loads = (PD * meta_cols + NUM_TCU_LANES - 1) / NUM_TCU_LANES;
  constexpr uint32_t per_k_tile_words = num_meta_loads * NUM_TCU_LANES;

  uint32_t num_k_tiles = d_attn / sp_ctx::tileK;
  uint32_t tile_row_idx = blockIdx.y;

  auto pMetaSp = pMetaSpBase + tile_row_idx * num_k_tiles * per_k_tile_words;
  auto pTileA = pA + tile_row * stride_A;
  constexpr uint32_t a_k_stride = sp_ctx::tileK / 2;

  auto pTileB = pB + tile_col * d_attn;
  for (int i = 0; i < (int)d_attn; i += (int)sp_ctx::tileK) {
    sp_ctx::load_matrix_sync<vt::row_major>(fragA, pTileA, stride_A, nullptr, pMetaSp);
    sp_ctx::load_matrix_sync<vt::col_major>(fragB, pTileB, d_attn);
    sp_ctx::mma_sync(fragC, fragA, fragB, fragC);
    pMetaSp += per_k_tile_words;
    pTileA  += a_k_stride;
    pTileB  += sp_ctx::tileK;
  }

  auto pTileC = pC + tile_row * N_attn + tile_col;
  sp_ctx::store_matrix_sync(pTileC, fragC, N_attn);
}

// Sparse TCU — O = P @ V with 2:4-compressed P (half K stride) + metadata.
// GEMM dims M=N_attn, N=d_attn, K=N_attn. Host prunes P post-softmax and
// stages the compressed buffer + packed metadata via P_addr/meta_P_addr.
void flash_pv_sparse(kernel_arg_t* __UNIFORM__ arg) {
  auto pA = reinterpret_cast<sp_ctx::input_t*>(arg->P_addr);
  auto pB = reinterpret_cast<sp_ctx::input_t*>(arg->V_addr);
  auto pC = reinterpret_cast<sp_ctx::output_t*>(arg->O_addr);
  auto pMetaSpBase = reinterpret_cast<const float*>(arg->meta_P_addr);

  uint32_t N_attn = arg->N;
  uint32_t d_attn = arg->d;
  uint32_t stride_A = N_attn / 2;

  sp_ctx::fragment_a   fragA;
  sp_ctx::fragment_b   fragB;
  sp_ctx::fragment_acc fragC;

  uint32_t tile_row = blockIdx.y * sp_ctx::tileM;
  uint32_t tile_col = blockIdx.x * sp_ctx::tileN;

  sp_ctx::fill_fragment(fragC, 0);

  constexpr uint32_t rtl_i_ratio = 32 / vt::fp16::bits;
  constexpr uint32_t meta_cols = (NUM_TCU_LANES * 2 * rtl_i_ratio + 31) / 32;
  using kcfg = vt::wmma_config_t<NUM_TCU_LANES>;
  constexpr uint32_t PD = kcfg::m_steps * (kcfg::k_steps / 2);
  constexpr uint32_t num_meta_loads = (PD * meta_cols + NUM_TCU_LANES - 1) / NUM_TCU_LANES;
  constexpr uint32_t per_k_tile_words = num_meta_loads * NUM_TCU_LANES;

  uint32_t num_k_tiles = N_attn / sp_ctx::tileK;
  uint32_t tile_row_idx = blockIdx.y;

  auto pMetaSp = pMetaSpBase + tile_row_idx * num_k_tiles * per_k_tile_words;
  auto pTileA = pA + tile_row * stride_A;
  constexpr uint32_t a_k_stride = sp_ctx::tileK / 2;

  auto pTileB = pB + tile_col * N_attn;
  for (int i = 0; i < (int)N_attn; i += (int)sp_ctx::tileK) {
    sp_ctx::load_matrix_sync<vt::row_major>(fragA, pTileA, stride_A, nullptr, pMetaSp);
    sp_ctx::load_matrix_sync<vt::col_major>(fragB, pTileB, N_attn);
    sp_ctx::mma_sync(fragC, fragA, fragB, fragC);
    pMetaSp += per_k_tile_words;
    pTileA  += a_k_stride;
    pTileB  += sp_ctx::tileK;
  }

  auto pTileC = pC + tile_row * d_attn + tile_col;
  sp_ctx::store_matrix_sync(pTileC, fragC, d_attn);
}

// Simple SIMT softmax — single thread per row.
void flash_softmax_body(kernel_arg_t* __UNIFORM__ arg) {
  auto S = reinterpret_cast<float*>(arg->S_addr);
  auto P = reinterpret_cast<float*>(arg->P_addr);
  uint32_t N = arg->N;
  int row = blockIdx.x;
  float max_val = S[row * N];
  for (uint32_t col = 1; col < N; ++col) {
    float v = S[row * N + col];
    if (v > max_val) max_val = v;
  }
  float local_P[256];
  float exp_sum = 0;
  for (uint32_t col = 0; col < N; ++col) {
    float e = expf(S[row * N + col] - max_val);
    local_P[col] = e;
    exp_sum += e;
  }
  for (uint32_t col = 0; col < N; ++col) P[row * N + col] = local_P[col] / exp_sum;
}

void flash_kernel_entry(kernel_arg_t* arg) {
  // Dispatch via kernel_id (set by host) for the unfused attention-style path.
  if (arg->kernel_id == KID_QK_SIMT)    { flash_qk_simt(arg);     return; }
  if (arg->kernel_id == KID_PV_SIMT)    { flash_pv_simt(arg);     return; }
  if (arg->kernel_id == KID_QK_TCU)     { flash_qk_tcu(arg);      return; }
  if (arg->kernel_id == KID_PV_TCU)     { flash_pv_tcu(arg);      return; }
  if (arg->kernel_id == KID_QK_SPARSE)  { flash_qk_sparse(arg);   return; }
  if (arg->kernel_id == KID_PV_SPARSE)  { flash_pv_sparse(arg);   return; }
  if (arg->kernel_id == KID_SOFTMAX)    { flash_softmax_body(arg);return; }

  if (arg->kernel_type == 1 && arg->head_dim == 8 && arg->block_size_r == 8 && arg->block_size_c == 8) {
    flashattention_tcu(arg);
    return;
  }

  switch (arg->head_dim) {
    case 1:
      switch (arg->block_size_c) {
        case 8:
          flash_kernel_body<1,8>(arg);
          return;
        case 16:
          flash_kernel_body<1,16>(arg);
          return;
        case 32:
          flash_kernel_body<1,32>(arg);
          return;
        case 64:
          flash_kernel_body<1,64>(arg);
          return;
        case 128:
          flash_kernel_body<1,128>(arg);
          return;
      }
      break;
    case 2:
      switch (arg->block_size_c) {
        case 8:
          flash_kernel_body<2,8>(arg);
          return;
        case 16:
          flash_kernel_body<2,16>(arg);
          return;
        case 32:
          flash_kernel_body<2,32>(arg);
          return;
        case 64:
          flash_kernel_body<2,64>(arg);
          return;    
      }
      break;
    case 4:
      switch (arg->block_size_c) {
        case 8:
          flash_kernel_body<4,8>(arg);
          return;
        case 16:
          flash_kernel_body<4,16>(arg);
          return;
        case 32:
          flash_kernel_body<4,32>(arg);
          return;
      }
      break;
    case 8:
      switch (arg->block_size_c) {
        case 8:
          flash_kernel_body<8,8>(arg);
          return;
        case 16:
          flash_kernel_body<8,16>(arg);
          return;
      }
      break;
    case 16:
      switch (arg->block_size_c) {
        case 8:
          flash_kernel_body<16,8>(arg);
          return;
      }
      break;
    }
}

int main() {
  kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
  uint64_t t_begin = vx_rdcycle();
  int rc = 0;
  switch (arg->kernel_id) {
    case KID_FLASH_FUSED:
      // Original fused flash-online-softmax SIMT path. Uses 1D grid with
      // blockDim.x threads per block; dispatches by head_dim/block_size_c
      // to the templated flash_kernel_body.
      rc = vx_spawn_threads(1, arg->grid_dim, arg->block_dim,
                            (vx_kernel_func_cb)flash_kernel_entry, arg);
      break;
    case KID_QK_SIMT:
      rc = vx_spawn_threads(2, arg->grid_dim, nullptr,
                            (vx_kernel_func_cb)flash_qk_simt, arg);
      break;
    case KID_SOFTMAX:
      rc = vx_spawn_threads(1, arg->grid_dim, nullptr,
                            (vx_kernel_func_cb)flash_softmax_body, arg);
      break;
    case KID_PV_SIMT:
      rc = vx_spawn_threads(2, arg->grid_dim, nullptr,
                            (vx_kernel_func_cb)flash_pv_simt, arg);
      break;
    case KID_QK_TCU:
      rc = vx_spawn_threads(2, arg->grid_dim, arg->block_dim,
                            (vx_kernel_func_cb)flash_qk_tcu, arg);
      break;
    case KID_PV_TCU:
      rc = vx_spawn_threads(2, arg->grid_dim, arg->block_dim,
                            (vx_kernel_func_cb)flash_pv_tcu, arg);
      break;
    case KID_QK_SPARSE:
      rc = vx_spawn_threads(2, arg->grid_dim, arg->block_dim,
                            (vx_kernel_func_cb)flash_qk_sparse, arg);
      break;
    case KID_PV_SPARSE:
      rc = vx_spawn_threads(2, arg->grid_dim, arg->block_dim,
                            (vx_kernel_func_cb)flash_pv_sparse, arg);
      break;
    default:
      return -1;
  }
  uint64_t t_end = vx_rdcycle();
  arg->kernel_cycles = t_end - t_begin;
  return rc;
}