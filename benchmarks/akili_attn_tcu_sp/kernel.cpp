#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_tensor.h>
#include "common.h"
#include <cmath>
#include <algorithm>

namespace vt = vortex::tensor;
using sp_ctx    = vt::wmma_context<NUM_TCU_LANES, vt::fp16, vt::fp32, true>;
using dense_ctx = vt::wmma_context<NUM_TCU_LANES, vt::fp16, vt::fp32, false>;

// Fast fp32 exp: avoids libm expf's softfloat-double fallback on rv32if.
// Branchless Taylor-then-square: u = x/32, Taylor(u), then y^32 via 5 squarings.
static inline float fast_exp(float x) {
  x = x < -80.0f ? -80.0f : x;
  float u = x * 0.03125f;
  float y = 1.0f + u*(1.0f + u*(0.5f + u*(0.166666667f + u*(0.041666667f + u*0.008333333f))));
  y = y*y; y = y*y; y = y*y; y = y*y; y = y*y;
  return y;
}

// =============================================================================
// Stage 2: SIMT softmax on fp32 S.
// =============================================================================
static inline void softmax_body(kernel_arg_t* arg) {
  auto S = reinterpret_cast<float*>(arg->S_addr);
  auto P = reinterpret_cast<float*>(arg->P_addr);
  uint32_t N = arg->N;

  uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= N) return;

  uint32_t hw_tid = blockIdx.x * NUM_THREADS + threadIdx.x;
  uint32_t stride = gridDim.x * NUM_THREADS;

  for (uint32_t row = hw_tid; row < N; row += stride) {
    float max_val = S[row * N];
    for (uint32_t col = 1; col < N; ++col)
      max_val = std::max(max_val, S[row * N + col]);

    float local_P[512];
    float exp_sum = 0;
    for (uint32_t col = 0; col < N; ++col) {
      float e = fast_exp(S[row * N + col] - max_val);
      local_P[col] = e;
      exp_sum += e;
    }
    for (uint32_t col = 0; col < N; ++col)
      P[row * N + col] = local_P[col] / exp_sum;
  }
}

// =============================================================================
// Sparse MMA inner loops, written in the exact sgemm_tcu_sp style: inlined into
// kernel_main, plain col-major B (no host pre-tiling), pTileB advanced with
// += tileK each iteration. This pattern has been verified to beat dense
// sgemm_tcu at every tested (NT, K) including K=32.
// =============================================================================
static inline void qk_sparse_body(kernel_arg_t* arg) {
  auto pA = reinterpret_cast<sp_ctx::input_t*>(arg->Q_addr);
  auto pB = reinterpret_cast<sp_ctx::input_t*>(arg->K_addr);
  auto pC = reinterpret_cast<sp_ctx::output_t*>(arg->S_addr);
  auto pMetaBase = reinterpret_cast<const float*>(arg->meta_Q_addr);
  uint32_t N_attn = arg->N;
  uint32_t K_walk = arg->d;

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
  uint32_t num_k_tiles = K_walk / sp_ctx::tileK;
  uint32_t stride_A = K_walk / 2;

  auto pMetaSp = pMetaBase + blockIdx.y * num_k_tiles * per_k_tile_words;
  auto pTileA = pA + tile_row * stride_A;
  auto pTileB = pB + tile_col * K_walk;
  constexpr uint32_t a_k_stride = sp_ctx::tileK / 2;

  for (int i = 0; i < (int)K_walk; i += (int)sp_ctx::tileK) {
    sp_ctx::load_matrix_sync<vt::row_major>(fragA, pTileA, stride_A, nullptr, pMetaSp);
    sp_ctx::load_matrix_sync<vt::col_major>(fragB, pTileB, K_walk);
    sp_ctx::mma_sync(fragC, fragA, fragB, fragC);
    pMetaSp += per_k_tile_words;
    pTileA  += a_k_stride;
    pTileB  += sp_ctx::tileK;
  }

  auto pTileC = pC + tile_row * N_attn + tile_col;
  sp_ctx::store_matrix_sync(pTileC, fragC, N_attn);
}

static inline void pv_sparse_body(kernel_arg_t* arg) {
  auto pA = reinterpret_cast<sp_ctx::input_t*>(arg->P_addr);
  auto pB = reinterpret_cast<sp_ctx::input_t*>(arg->V_addr);
  auto pC = reinterpret_cast<sp_ctx::output_t*>(arg->O_addr);
  auto pMetaBase = reinterpret_cast<const float*>(arg->meta_P_addr);
  uint32_t N_attn = arg->N;
  uint32_t d_attn = arg->d;
  uint32_t K_walk = N_attn;

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
  uint32_t num_k_tiles = K_walk / sp_ctx::tileK;
  uint32_t stride_A = K_walk / 2;

  auto pMetaSp = pMetaBase + blockIdx.y * num_k_tiles * per_k_tile_words;
  auto pTileA = pA + tile_row * stride_A;
  auto pTileB = pB + tile_col * K_walk;
  constexpr uint32_t a_k_stride = sp_ctx::tileK / 2;

  for (int i = 0; i < (int)K_walk; i += (int)sp_ctx::tileK) {
    sp_ctx::load_matrix_sync<vt::row_major>(fragA, pTileA, stride_A, nullptr, pMetaSp);
    sp_ctx::load_matrix_sync<vt::col_major>(fragB, pTileB, K_walk);
    sp_ctx::mma_sync(fragC, fragA, fragB, fragC);
    pMetaSp += per_k_tile_words;
    pTileA  += a_k_stride;
    pTileB  += sp_ctx::tileK;
  }

  auto pTileC = pC + tile_row * d_attn + tile_col;
  sp_ctx::store_matrix_sync(pTileC, fragC, d_attn);
}

// S2: dense PV path for hybrid sparse-QK + dense-PV attention.
static inline void pv_dense_body(kernel_arg_t* arg) {
  auto pA = reinterpret_cast<dense_ctx::input_t*>(arg->P_addr);
  auto pB = reinterpret_cast<dense_ctx::input_t*>(arg->V_addr);
  auto pC = reinterpret_cast<dense_ctx::output_t*>(arg->O_addr);
  uint32_t N_attn = arg->N;
  uint32_t d_attn = arg->d;

  dense_ctx::fragment_a   fragA;
  dense_ctx::fragment_b   fragB;
  dense_ctx::fragment_acc fragC;
  uint32_t tile_row = blockIdx.y * dense_ctx::tileM;
  uint32_t tile_col = blockIdx.x * dense_ctx::tileN;
  dense_ctx::fill_fragment(fragC, 0);

  for (int i = 0; i < (int)N_attn; i += (int)dense_ctx::tileK) {
    auto pTileA = pA + tile_row * N_attn + i;
    auto pTileB = pB + tile_col * N_attn + i;
    dense_ctx::load_matrix_sync(fragA, pTileA, N_attn);
    dense_ctx::load_matrix_sync<vt::col_major>(fragB, pTileB, N_attn);
    dense_ctx::mma_sync(fragC, fragA, fragB, fragC);
  }
  auto pTileC = pC + tile_row * d_attn + tile_col;
  dense_ctx::store_matrix_sync(pTileC, fragC, d_attn);
}

// =============================================================================
__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  __rdcycle_time t0 = vx_rdcycle_sync_begin();
  uint32_t i0 = csr_read(VX_CSR_MINSTRET);

  switch (arg->kernel_id) {
    case KID_QK_SPARSE: qk_sparse_body(arg); break;
    case KID_SOFTMAX:   softmax_body(arg);   break;
    case KID_PV_SPARSE: pv_sparse_body(arg); break;
    case KID_PV_DENSE:  pv_dense_body(arg);  break;
    default: break;
  }

  uint32_t i1 = csr_read(VX_CSR_MINSTRET);
  __rdcycle_time t1 = vx_rdcycle_sync_end();
  if (threadIdx.x == 0) {
    auto pCycles = reinterpret_cast<uint32_t*>(arg->cycles_addr);
    auto pInstrs = reinterpret_cast<uint32_t*>(arg->instrs_addr);
    uint32_t block_id = blockIdx.y * gridDim.x + blockIdx.x;
    pCycles[block_id] = (uint32_t)vx_rdcycle_sync_diff(t0, t1);
    pInstrs[block_id] = i1 - i0;
  }
}
