#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_tensor.h>
#include "common.h"
#include <cmath>
#include <algorithm>

namespace vt = vortex::tensor;
using tcu_ctx = vt::wmma_context<NUM_TCU_LANES, vt::fp16, vt::fp32, false>;

// Fast fp32 exp: avoids libm expf's softfloat-double fallback on rv32if.
// Branchless: clamp x to [-80, 0-ish], Taylor around u = x/32 (converges fast),
// then square 5 times to get exp(x). Pure fp32, no union/bitcast, no branches
// (beyond one fminf/fmaxf) — safe for Vortex's divergent-aware codegen.
static inline float fast_exp(float x) {
  x = x < -80.0f ? -80.0f : x;
  float u = x * 0.03125f;  // x / 32
  float y = 1.0f + u*(1.0f + u*(0.5f + u*(0.166666667f + u*(0.041666667f + u*0.008333333f))));
  y = y*y; y = y*y; y = y*y; y = y*y; y = y*y;  // y^32
  return y;
}

// =============================================================================
// Stage 2: SIMT softmax on fp32 S. One row per flat thread index.
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
// Stage 1: Dense TCU S = Q·Kᵀ  — one output tile per 2D block.
// =============================================================================
static inline void qk_tcu_body(kernel_arg_t* arg) {
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

// =============================================================================
// Stage 3: Dense TCU O = P·V
// =============================================================================
static inline void pv_tcu_body(kernel_arg_t* arg) {
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

// =============================================================================
__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  __rdcycle_time t0 = vx_rdcycle_sync_begin();
  uint32_t i0 = csr_read(VX_CSR_MINSTRET);

  switch (arg->kernel_id) {
    case KID_QK_TCU:  qk_tcu_body(arg);  break;
    case KID_SOFTMAX: softmax_body(arg); break;
    case KID_PV_TCU:  pv_tcu_body(arg);  break;
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
