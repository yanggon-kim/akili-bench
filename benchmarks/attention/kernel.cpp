#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include "common.h"
#include <cmath>
#include <algorithm>
#include <cstring>

#ifdef ENABLE_TCU
#include <vx_tensor.h>
namespace vt = vortex::tensor;
// Dense fp16->fp32 context sized to the current warp width (NUM_TCU_LANES).
using tcu_ctx = vt::wmma_context<NUM_TCU_LANES, vt::fp16, vt::fp32, false>;
using sp_ctx  = vt::wmma_context<NUM_TCU_LANES, vt::fp16, vt::fp32, true>;
static constexpr uint32_t TM = tcu_ctx::tileM;
static constexpr uint32_t TN = tcu_ctx::tileN;
static constexpr uint32_t TK = tcu_ctx::tileK;  // in fp16 elements
#endif

// =============================================================================
// SIMT path — one thread per output cell (NT-agnostic: no block_dim)
// =============================================================================

void kernel0_body(kernel_arg_t* __UNIFORM__ arg) {
  auto Q = reinterpret_cast<TYPE*>(arg->Q_addr);
  auto K = reinterpret_cast<TYPE*>(arg->K_addr);
  auto S = reinterpret_cast<TYPE*>(arg->S_addr);
  auto N = arg->N;
  auto d = arg->d;

  int col = blockIdx.x;
  int row = blockIdx.y;

  if (row < (int)N && col < (int)N) {
    TYPE sum(0);
    for (uint32_t e = 0; e < d; ++e) {
      sum += Q[row * d + e] * K[e * N + col];
    }
    S[row * N + col] = sum;
  }
}

void kernel1_body(kernel_arg_t* __UNIFORM__ arg) {
  auto S = reinterpret_cast<TYPE*>(arg->S_addr);
  auto P = reinterpret_cast<TYPE*>(arg->P_addr);
  auto N = arg->N;

  int row = blockIdx.x;

  TYPE max_val = S[row * N];
  for (uint32_t col = 1; col < N; ++col) {
    max_val = std::max(max_val, S[row * N + col]);
  }

  TYPE local_P[256];  // oversized to cover padded TCU defaults
  TYPE exp_sum = 0;
  for (uint32_t col = 0; col < N; ++col) {
    auto exp = std::exp(S[row * N + col] - max_val);
    local_P[col] = exp;
    exp_sum += exp;
  }

  for (uint32_t col = 0; col < N; ++col) {
    P[row * N + col] = local_P[col] / exp_sum;
  }
}

void kernel2_body(kernel_arg_t* __UNIFORM__ arg) {
  auto P = reinterpret_cast<TYPE*>(arg->P_addr);
  auto V = reinterpret_cast<TYPE*>(arg->V_addr);
  auto O = reinterpret_cast<TYPE*>(arg->O_addr);
  auto N = arg->N;
  auto d = arg->d;

  int col = blockIdx.x;
  int row = blockIdx.y;

  if (row < (int)N && col < (int)d) {
    TYPE sum(0);
    for (uint32_t e = 0; e < N; ++e) {
      sum += P[row * N + e] * V[e * d + col];
    }
    O[row * d + col] = sum;
  }
}

#ifdef ENABLE_TCU
// =============================================================================
// TCU kernels — refactored to mirror tests/regression/sgemm_tcu verbatim.
// One output tile per block, direct global→fragment load, no local memory
// staging, no device-side fp32→fp16 conversion. Host pre-packs Q/K/V/P as
// fp16 and transposes K, V so a single `load_matrix_sync<col_major>(fragB,
// pTileB, K)` can read the B columns directly.
//
// Grid layout:  grid_dim = (N_cols / tileN, M_rows / tileM)
// Block layout: block_dim = (NUM_THREADS, 1)  (one warp per block)
// =============================================================================

// Dense TCU — S = Q @ Kᵀ  (GEMM M=N_attn, N=N_attn, K=d_attn).
// Host packs Q as fp16 row-major [N × d] and K as fp16 col-major [N × d].
void kernel3_body(kernel_arg_t* __UNIFORM__ arg) {
  auto pA = reinterpret_cast<tcu_ctx::input_t*>(arg->Q_addr);
  auto pB = reinterpret_cast<tcu_ctx::input_t*>(arg->K_addr);
  auto pC = reinterpret_cast<tcu_ctx::output_t*>(arg->S_addr);

  uint32_t N_attn = arg->N;   // output rows/cols
  uint32_t d_attn = arg->d;   // inner K

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

// Dense TCU — O = P @ V  (GEMM M=N_attn, N=d_attn, K=N_attn).
// Host packs P as fp16 row-major [N × N] and V as fp16 col-major [d × N].
void kernel4_body(kernel_arg_t* __UNIFORM__ arg) {
  auto pA = reinterpret_cast<tcu_ctx::input_t*>(arg->P_addr);
  auto pB = reinterpret_cast<tcu_ctx::input_t*>(arg->V_addr);
  auto pC = reinterpret_cast<tcu_ctx::output_t*>(arg->O_addr);

  uint32_t N_attn = arg->N;   // inner K for P@V
  uint32_t d_attn = arg->d;   // output cols

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

// Sparse TCU — S = Q @ Kᵀ  with 2:4-compressed Q (half K stride) + metadata.
// Mirrors tests/regression/sgemm_tcu_sp/kernel.cpp verbatim.
void kernel3_sparse_body(kernel_arg_t* __UNIFORM__ arg) {
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

  // Per-K-tile metadata — same formula as sgemm_tcu_sp.
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

// Sparse TCU — O = P @ V  with 2:4-compressed P (half K stride) + metadata.
// GEMM dims are M=N_attn, N=d_attn, K=N_attn. Mirrors kernel3_sparse_body with
// P taking the role of the compressed-A matrix. The host prunes and packs P
// between the softmax kernel and this kernel.
void kernel4_sparse_body(kernel_arg_t* __UNIFORM__ arg) {
  auto pA = reinterpret_cast<sp_ctx::input_t*>(arg->P_addr);       // compressed fp16 P
  auto pB = reinterpret_cast<sp_ctx::input_t*>(arg->V_addr);       // fp16 V (col-major)
  auto pC = reinterpret_cast<sp_ctx::output_t*>(arg->O_addr);
  auto pMetaSpBase = reinterpret_cast<const float*>(arg->meta_P_addr);

  uint32_t N_attn = arg->N;
  uint32_t d_attn = arg->d;
  uint32_t stride_A = N_attn / 2;   // compressed-P row stride

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
#endif // ENABLE_TCU

// =============================================================================
// Entry point — wraps each spawn in WSYNC cycle timing
// =============================================================================

int main() {
  kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);

  // Timing: read the 64-bit cycle counter at entry / exit. No WSYNC wrapper,
  // to rule it out as a source of interference with the kernel body.
  uint64_t t_begin = vx_rdcycle();
  int rc = 0;

  switch (arg->kernel_id) {
    case KID_QK_SIMT:
      rc = vx_spawn_threads(2, arg->grid_dim, nullptr,
                            (vx_kernel_func_cb)kernel0_body, arg);
      break;
    case KID_SOFTMAX:
      rc = vx_spawn_threads(1, arg->grid_dim, nullptr,
                            (vx_kernel_func_cb)kernel1_body, arg);
      break;
    case KID_PV_SIMT:
      rc = vx_spawn_threads(2, arg->grid_dim, nullptr,
                            (vx_kernel_func_cb)kernel2_body, arg);
      break;
#ifdef ENABLE_TCU
    case KID_QK_TCU:
      rc = vx_spawn_threads(2, arg->grid_dim, arg->block_dim,
                            (vx_kernel_func_cb)kernel3_body, arg);
      break;
    case KID_PV_TCU:
      rc = vx_spawn_threads(2, arg->grid_dim, arg->block_dim,
                            (vx_kernel_func_cb)kernel4_body, arg);
      break;
    case KID_QK_SPARSE:
      rc = vx_spawn_threads(2, arg->grid_dim, arg->block_dim,
                            (vx_kernel_func_cb)kernel3_sparse_body, arg);
      break;
    case KID_PV_SPARSE:
      // Host prunes P to 2:4 after softmax and passes the compressed buffer
      // via P_addr + meta_P_addr. Sparse TCU runs both QK and PV stages.
      rc = vx_spawn_threads(2, arg->grid_dim, arg->block_dim,
                            (vx_kernel_func_cb)kernel4_sparse_body, arg);
      break;
#endif
    default:
      return -1;
  }

  uint64_t t_end = vx_rdcycle();
  arg->kernel_cycles = t_end - t_begin;
  return rc;
}
