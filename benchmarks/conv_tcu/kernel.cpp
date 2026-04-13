#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include "common.h"

#ifdef ENABLE_TCU
#include <vx_tensor.h>
namespace vt = vortex::tensor;
using tcu_ctx = vt::wmma_context<NUM_TCU_LANES, vt::fp16, vt::fp32, false>;
using sp_ctx  = vt::wmma_context<NUM_TCU_LANES, vt::fp16, vt::fp32, true>;
#endif

// =============================================================================
// SIMT conv — one thread per output element via 2D (oc, spatial_idx) grid.
// Each thread computes one output point by directly looping over C_in × K × K.
// =============================================================================
void kernel_conv_simt(kernel_arg_t* __UNIFORM__ arg) {
  auto I = reinterpret_cast<float*>(arg->I_addr);
  auto W = reinterpret_cast<float*>(arg->W_addr);
  auto O = reinterpret_cast<float*>(arg->O_addr);

  uint32_t C_in  = arg->C_in;
  uint32_t H     = arg->H;
  uint32_t Wd    = arg->W;
  uint32_t K     = arg->K_sz;
  uint32_t H_out = arg->H_out;
  uint32_t W_out = arg->W_out;

  int oc     = blockIdx.y;      // output channel
  int sp_idx = blockIdx.x;      // flattened (oy, ox)
  int oy = sp_idx / (int)W_out;
  int ox = sp_idx % (int)W_out;

  if (oy >= (int)H_out || ox >= (int)W_out) return;

  float sum = 0.0f;
  uint32_t wt_base = oc * C_in * K * K;
  for (uint32_t ic = 0; ic < C_in; ++ic) {
    for (uint32_t ky = 0; ky < K; ++ky) {
      for (uint32_t kx = 0; kx < K; ++kx) {
        int in_y = oy + (int)ky;
        int in_x = ox + (int)kx;
        if (in_y >= 0 && in_y < (int)H && in_x >= 0 && in_x < (int)Wd) {
          uint32_t in_idx = ic * H * Wd + in_y * Wd + in_x;
          uint32_t wt_idx = wt_base + (ic * K + ky) * K + kx;
          sum += I[in_idx] * W[wt_idx];
        }
      }
    }
  }
  O[oc * H_out * W_out + oy * W_out + ox] = sum;
}

#ifdef ENABLE_TCU
// =============================================================================
// Dense TCU conv — mirrors sgemm_tcu/kernel.cpp. One output tile per block.
// A = W_gemm [M_gemm × K_gemm] fp16 row-major
// B = Icol   [N_gemm × K_gemm] fp16 col-major (host-prepped via im2col)
// C = Out    [M_gemm × N_gemm] fp32 row-major
// =============================================================================
void kernel_conv_tcu_body(kernel_arg_t* __UNIFORM__ arg) {
  auto pA = reinterpret_cast<tcu_ctx::input_t*>(arg->W_addr);
  auto pB = reinterpret_cast<tcu_ctx::input_t*>(arg->B_addr);
  auto pC = reinterpret_cast<tcu_ctx::output_t*>(arg->O_addr);

  uint32_t M = arg->M_gemm;
  uint32_t N = arg->N_gemm;
  uint32_t K = arg->K_gemm;

  tcu_ctx::fragment_a   fragA;
  tcu_ctx::fragment_b   fragB;
  tcu_ctx::fragment_acc fragC;

  uint32_t tile_row = blockIdx.y * tcu_ctx::tileM;
  uint32_t tile_col = blockIdx.x * tcu_ctx::tileN;

  tcu_ctx::fill_fragment(fragC, 0);

  for (int i = 0; i < (int)K; i += (int)tcu_ctx::tileK) {
    auto pTileA = pA + tile_row * K + i;
    auto pTileB = pB + tile_col * K + i;
    tcu_ctx::load_matrix_sync(fragA, pTileA, K);
    tcu_ctx::load_matrix_sync<vt::col_major>(fragB, pTileB, K);
    tcu_ctx::mma_sync(fragC, fragA, fragB, fragC);
  }

  auto pTileC = pC + tile_row * N + tile_col;
  tcu_ctx::store_matrix_sync(pTileC, fragC, N);
}

// =============================================================================
// Sparse TCU conv — host prunes W 2:4 along K_gemm, compresses to half stride,
// packs metadata. Device kernel mirrors sgemm_tcu_sp/kernel.cpp.
// =============================================================================
void kernel_conv_sparse_body(kernel_arg_t* __UNIFORM__ arg) {
  auto pA = reinterpret_cast<sp_ctx::input_t*>(arg->W_addr);   // compressed fp16 W
  auto pB = reinterpret_cast<sp_ctx::input_t*>(arg->B_addr);   // fp16 Icol (col-major)
  auto pC = reinterpret_cast<sp_ctx::output_t*>(arg->O_addr);
  auto pMetaSpBase = reinterpret_cast<const float*>(arg->meta_W_addr);

  uint32_t M = arg->M_gemm;
  uint32_t N = arg->N_gemm;
  uint32_t K = arg->K_gemm;
  uint32_t stride_A = K / 2;

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

  uint32_t num_k_tiles = K / sp_ctx::tileK;
  uint32_t tile_row_idx = blockIdx.y;

  auto pMetaSp = pMetaSpBase + tile_row_idx * num_k_tiles * per_k_tile_words;
  auto pTileA = pA + tile_row * stride_A;
  constexpr uint32_t a_k_stride = sp_ctx::tileK / 2;

  auto pTileB = pB + tile_col * K;
  for (int i = 0; i < (int)K; i += (int)sp_ctx::tileK) {
    sp_ctx::load_matrix_sync<vt::row_major>(fragA, pTileA, stride_A, nullptr, pMetaSp);
    sp_ctx::load_matrix_sync<vt::col_major>(fragB, pTileB, K);
    sp_ctx::mma_sync(fragC, fragA, fragB, fragC);
    pMetaSp += per_k_tile_words;
    pTileA  += a_k_stride;
    pTileB  += sp_ctx::tileK;
  }

  auto pTileC = pC + tile_row * N + tile_col;
  sp_ctx::store_matrix_sync(pTileC, fragC, N);
}
#endif // ENABLE_TCU

// =============================================================================
// Entry point — wraps spawn in WSYNC cycle timing.
// =============================================================================
int main() {
  kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);

  uint64_t t_begin = vx_rdcycle();
  int rc = 0;

  switch (arg->kernel_id) {
    case KID_CONV_SIMT:
      rc = vx_spawn_threads(2, arg->grid_dim, nullptr,
                            (vx_kernel_func_cb)kernel_conv_simt, arg);
      break;
#ifdef ENABLE_TCU
    case KID_CONV_TCU:
      rc = vx_spawn_threads(2, arg->grid_dim, arg->block_dim,
                            (vx_kernel_func_cb)kernel_conv_tcu_body, arg);
      break;
    case KID_CONV_SPARSE:
      rc = vx_spawn_threads(2, arg->grid_dim, arg->block_dim,
                            (vx_kernel_func_cb)kernel_conv_sparse_body, arg);
      break;
#endif
    default:
      return -1;
  }

  uint64_t t_end = vx_rdcycle();
  arg->kernel_cycles = t_end - t_begin;
  return rc;
}
