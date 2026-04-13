#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include <vx_tensor.h>
#include "common.h"
#include <cmath>
#include <algorithm>

namespace vt = vortex::tensor;
using tcu_ctx = vt::wmma_context<NUM_TCU_LANES, vt::fp16, vt::fp32, false>;

// =============================================================================
// Stage 2: SIMT softmax on fp32 S. One thread per row.
// =============================================================================
void kernel_softmax(kernel_arg_t* __UNIFORM__ arg) {
  auto S = reinterpret_cast<float*>(arg->S_addr);
  auto P = reinterpret_cast<float*>(arg->P_addr);
  uint32_t N = arg->N;
  int row = blockIdx.x;

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

// =============================================================================
// Stage 1: Dense TCU S = Q·Kᵀ   (M=N_attn, N=N_attn, K=d_attn)
// Mirrors tests/regression/sgemm_tcu/kernel.cpp verbatim.
// =============================================================================
void kernel_qk_tcu(kernel_arg_t* __UNIFORM__ arg) {
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
// Stage 3: Dense TCU O = P·V   (M=N_attn, N=d_attn, K=N_attn)
// =============================================================================
void kernel_pv_tcu(kernel_arg_t* __UNIFORM__ arg) {
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
// Entry point
// =============================================================================
int main() {
  kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
  uint64_t t_begin = vx_rdcycle();
  int rc = 0;

  switch (arg->kernel_id) {
    case KID_QK_TCU:
      rc = vx_spawn_threads(2, arg->grid_dim, arg->block_dim,
                            (vx_kernel_func_cb)kernel_qk_tcu, arg);
      break;
    case KID_SOFTMAX:
      rc = vx_spawn_threads(1, arg->grid_dim, nullptr,
                            (vx_kernel_func_cb)kernel_softmax, arg);
      break;
    case KID_PV_TCU:
      rc = vx_spawn_threads(2, arg->grid_dim, arg->block_dim,
                            (vx_kernel_func_cb)kernel_pv_tcu, arg);
      break;
    default:
      return -1;
  }

  uint64_t t_end = vx_rdcycle();
  arg->kernel_cycles = t_end - t_begin;
  return rc;
}
