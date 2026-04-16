#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_tensor.h>
#include "common.h"
#include <cmath>
#include <algorithm>

namespace vt = vortex::tensor;
using sp_ctx = vt::wmma_context<NUM_TCU_LANES, vt::fp16, vt::fp32, true>;

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
      float e = std::exp(S[row * N + col] - max_val);
      local_P[col] = e;
      exp_sum += e;
    }
    for (uint32_t col = 0; col < N; ++col)
      P[row * N + col] = local_P[col] / exp_sum;
  }
}

// =============================================================================
// Shared sparse MMA inner loop.
// =============================================================================
static inline void sparse_mma_loop(sp_ctx::input_t* pA_base,
                                   sp_ctx::input_t* pB_base,
                                   sp_ctx::output_t* pC_base,
                                   const float* pMetaBase,
                                   uint32_t K_walk,
                                   uint32_t c_stride) {
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
  uint32_t tile_row_idx = blockIdx.y;
  uint32_t stride_A = K_walk / 2;

  auto pMetaSp = pMetaBase + tile_row_idx * num_k_tiles * per_k_tile_words;
  auto pTileA = pA_base + tile_row * stride_A;
  constexpr uint32_t a_k_stride = sp_ctx::tileK / 2;
  // B is pre-tiled by the host into contiguous tileK×tileN col-major blocks.
  // Block(n_tile, k_tile) is at offset (n_tile * num_k_tiles + k_tile) * block_elems.
  // Stride within each block = tileK (not K_walk), giving tight cache reads.
  constexpr uint32_t block_elems = sp_ctx::tileK * sp_ctx::tileN;
  uint32_t n_tile = blockIdx.x;

  for (int i = 0; i < (int)K_walk; i += (int)sp_ctx::tileK) {
    uint32_t k_tile = (uint32_t)i / sp_ctx::tileK;
    auto pTileB = pB_base + (n_tile * num_k_tiles + k_tile) * block_elems;
    sp_ctx::load_matrix_sync<vt::row_major>(fragA, pTileA, stride_A, nullptr, pMetaSp);
    sp_ctx::load_matrix_sync<vt::col_major>(fragB, pTileB, sp_ctx::tileK);
    sp_ctx::mma_sync(fragC, fragA, fragB, fragC);
    pMetaSp += per_k_tile_words;
    pTileA  += a_k_stride;
  }

  auto pTileC = pC_base + tile_row * c_stride + tile_col;
  sp_ctx::store_matrix_sync(pTileC, fragC, c_stride);
}

static inline void qk_sparse_body(kernel_arg_t* arg) {
  auto pA = reinterpret_cast<sp_ctx::input_t*>(arg->Q_addr);
  auto pB = reinterpret_cast<sp_ctx::input_t*>(arg->K_addr);
  auto pC = reinterpret_cast<sp_ctx::output_t*>(arg->S_addr);
  auto pMetaBase = reinterpret_cast<const float*>(arg->meta_Q_addr);
  sparse_mma_loop(pA, pB, pC, pMetaBase, arg->d, arg->N);
}

static inline void pv_sparse_body(kernel_arg_t* arg) {
  auto pA = reinterpret_cast<sp_ctx::input_t*>(arg->P_addr);
  auto pB = reinterpret_cast<sp_ctx::input_t*>(arg->V_addr);
  auto pC = reinterpret_cast<sp_ctx::output_t*>(arg->O_addr);
  auto pMetaBase = reinterpret_cast<const float*>(arg->meta_P_addr);
  sparse_mma_loop(pA, pB, pC, pMetaBase, arg->N, arg->d);
}

// =============================================================================
__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  __rdcycle_time t0 = vx_rdcycle_sync_begin();

  switch (arg->kernel_id) {
    case KID_QK_SPARSE: qk_sparse_body(arg); break;
    case KID_SOFTMAX:   softmax_body(arg);   break;
    case KID_PV_SPARSE: pv_sparse_body(arg); break;
    default: break;
  }

  __rdcycle_time t1 = vx_rdcycle_sync_end();
  if (threadIdx.x == 0) {
    auto pCycles = reinterpret_cast<uint32_t*>(arg->cycles_addr);
    uint32_t block_id = blockIdx.y * gridDim.x + blockIdx.x;
    pCycles[block_id] = (uint32_t)vx_rdcycle_sync_diff(t0, t1);
  }
}
