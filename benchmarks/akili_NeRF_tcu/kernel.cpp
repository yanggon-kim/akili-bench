#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include <vx_tensor.h>
#include <math.h>
#include "common.h"

// =============================================================================
// akili_NeRF_tcu — Full NeRF forward pass with TCU-accelerated MLP.
//
//   Stage 1 (SIMT):  ray-AABB slab + stratified sampling + positional encoding
//   Stage 2 (TCU):   4 layer GEMMs (W x X -> Y_fp32)   — dispatched 4x
//   Stage 2 (SIMT):  ReLU + bias + fp32->fp16 cast     — dispatched 4x
//                    (last layer uses KID_MLP_OUT_ACT = softplus+sigmoid)
//   Stage 3 (SIMT):  per-ray alpha compositing
// =============================================================================

namespace vt = vortex::tensor;
using tcu_ctx = vt::wmma_context<NUM_TCU_LANES, vt::fp16, vt::fp32, false>;

// -----------------------------------------------------------------------------
// Stage 1: ray setup + positional encoding.
// One task per POINT (flat n_rays*n_samples tasks) so the strided writes to
// pe_buf are cache-friendly: within each warp, lane l writes feature f for
// point (warp_base + l) — 8 contiguous fp16 values per cache line.
// Writes PE features into pe_buf in [32 x n_points] row-major layout.
// -----------------------------------------------------------------------------
void kernel_ray_setup(kernel_arg_t* __UNIFORM__ arg) {
  auto rays_o  = reinterpret_cast<float*>(arg->rays_o_addr);
  auto rays_d  = reinterpret_cast<float*>(arg->rays_d_addr);
  auto aabb    = reinterpret_cast<float*>(arg->aabb_addr);
  auto pe_out  = reinterpret_cast<uint16_t*>(arg->pe_buf_addr);
  auto dlt_out = reinterpret_cast<float*>(arg->deltas_addr);

  uint32_t pt     = blockIdx.x;                // flat point index
  uint32_t S      = arg->n_samples;
  uint32_t n_pts  = arg->n_points;
  uint32_t ray    = pt / S;
  uint32_t s      = pt - ray * S;
  float min_near  = arg->min_near;

  float ox = rays_o[ray*3+0], oy = rays_o[ray*3+1], oz = rays_o[ray*3+2];
  float dx = rays_d[ray*3+0], dy = rays_d[ray*3+1], dz = rays_d[ray*3+2];

  float inv_dx = 1.0f / dx, inv_dy = 1.0f / dy, inv_dz = 1.0f / dz;
  float tx1 = (aabb[0] - ox) * inv_dx, tx2 = (aabb[3] - ox) * inv_dx;
  float ty1 = (aabb[1] - oy) * inv_dy, ty2 = (aabb[4] - oy) * inv_dy;
  float tz1 = (aabb[2] - oz) * inv_dz, tz2 = (aabb[5] - oz) * inv_dz;
  float tmin_x = (tx1 < tx2) ? tx1 : tx2;
  float tmax_x = (tx1 > tx2) ? tx1 : tx2;
  float tmin_y = (ty1 < ty2) ? ty1 : ty2;
  float tmax_y = (ty1 > ty2) ? ty1 : ty2;
  float tmin_z = (tz1 < tz2) ? tz1 : tz2;
  float tmax_z = (tz1 > tz2) ? tz1 : tz2;
  float tmin = tmin_x;
  if (tmin_y > tmin) tmin = tmin_y;
  if (tmin_z > tmin) tmin = tmin_z;
  float tmax = tmax_x;
  if (tmax_y < tmax) tmax = tmax_y;
  if (tmax_z < tmax) tmax = tmax_z;
  if (tmin < min_near) tmin = min_near;
  float step = 0.0f;
  if (tmax > tmin) step = (tmax - tmin) / (float)S;

  float t  = tmin + ((float)s + 0.5f) * step;
  float px = ox + t * dx;
  float py = oy + t * dy;
  float pz = oz + t * dz;
  dlt_out[pt] = step;

  // Device-side fp32 -> fp16 (IEEE) packing. Same bit tricks as f2h_host.
  auto f2h = [](float x) -> uint16_t {
    uint32_t bits;
    __builtin_memcpy(&bits, &x, sizeof(bits));
    uint16_t sign = (uint16_t)((bits >> 16) & 0x8000);
    uint16_t mant = (uint16_t)((bits >> 13) & 0x03FF);
    int32_t  ex   = (int32_t)((bits >> 23) & 0xFF) - 127 + 15;
    if (ex >= 31) return (uint16_t)(sign | 0x7C00);
    if (ex <= 0)  return sign;
    return (uint16_t)(sign | ((uint16_t)ex << 10) | mant);
  };

  // Feature-major layout: pe_out[f * n_pts + pt].
  // Within a warp, consecutive lanes write consecutive `pt` at the same
  // feature row → 8 contiguous fp16 = 16 bytes into one cache line.
  pe_out[0 * n_pts + pt] = f2h(px);
  pe_out[1 * n_pts + pt] = f2h(py);
  pe_out[2 * n_pts + pt] = f2h(pz);
  float freq = 1.0f;
  for (int L = 0; L < PE_L; ++L) {
    uint32_t base = 3 + L*6;
    pe_out[(base+0) * n_pts + pt] = f2h(sinf(freq * px));
    pe_out[(base+1) * n_pts + pt] = f2h(cosf(freq * px));
    pe_out[(base+2) * n_pts + pt] = f2h(sinf(freq * py));
    pe_out[(base+3) * n_pts + pt] = f2h(cosf(freq * py));
    pe_out[(base+4) * n_pts + pt] = f2h(sinf(freq * pz));
    pe_out[(base+5) * n_pts + pt] = f2h(cosf(freq * pz));
    freq *= 2.0f;
  }
  // Pad feature rows 27..31 with zero so layer 0's K-loop is clean.
  for (uint32_t f = MLP_IN_DIM; f < MLP_IN_DIM_PAD; ++f)
    pe_out[f * n_pts + pt] = 0;
}

// -----------------------------------------------------------------------------
// Stage 2: dense TCU MLP layer (one 2D grid block per output tile).
//   C = A * B  where:
//     A = W [N_out x K_in] row-major  (default row-major load)
//     B = X [K_in x n_points] row-major  (default row-major load)
//     C = Y [N_out x n_points] row-major store
// Grid: (n_points / tileN, N_out / tileM)
// -----------------------------------------------------------------------------
void kernel_mlp_gemm(kernel_arg_t* __UNIFORM__ arg) {
  auto pA = reinterpret_cast<tcu_ctx::input_t*>(arg->W_cur_addr);
  auto pB = reinterpret_cast<tcu_ctx::input_t*>(arg->X_cur_addr);
  auto pC = reinterpret_cast<tcu_ctx::output_t*>(arg->Y_cur_addr);

  uint32_t K_in  = arg->K_in_cur;
  uint32_t n_pts = arg->n_points;

  tcu_ctx::fragment_a   fragA;
  tcu_ctx::fragment_b   fragB;
  tcu_ctx::fragment_acc fragC;

  uint32_t tile_row = blockIdx.y * tcu_ctx::tileM;  // output row start
  uint32_t tile_col = blockIdx.x * tcu_ctx::tileN;  // point col start
  tcu_ctx::fill_fragment(fragC, 0);

  for (int i = 0; i < (int)K_in; i += (int)tcu_ctx::tileK) {
    auto pTileA = pA + tile_row * K_in + i;
    auto pTileB = pB + i * n_pts + tile_col;
    tcu_ctx::load_matrix_sync(fragA, pTileA, K_in);
    tcu_ctx::load_matrix_sync(fragB, pTileB, n_pts);
    tcu_ctx::mma_sync(fragC, fragA, fragB, fragC);
  }

  auto pTileC = pC + tile_row * n_pts + tile_col;
  tcu_ctx::store_matrix_sync(pTileC, fragC, n_pts);
}

// -----------------------------------------------------------------------------
// Stage 2: SIMT activation + bias + fp32->fp16 cast for hidden layers.
// One task per (n_out_row, point) flat index. Grid = N_out * n_points.
// -----------------------------------------------------------------------------
void kernel_mlp_act(kernel_arg_t* __UNIFORM__ arg) {
  auto pY_fp32 = reinterpret_cast<float*>(arg->Y_cur_addr);
  auto pB      = reinterpret_cast<float*>(arg->B_cur_addr);
  auto pOut    = reinterpret_cast<uint16_t*>(arg->act_out_addr);

  uint32_t tid = blockIdx.x;
  uint32_t n_pts = arg->n_points;
  uint32_t row = tid / n_pts;
  uint32_t col = tid - row * n_pts;

  float v = pY_fp32[row * n_pts + col] + pB[row];
  if (v < 0.0f) v = 0.0f;

  // fp32 -> fp16
  uint32_t bits;
  __builtin_memcpy(&bits, &v, sizeof(bits));
  uint16_t sign = (uint16_t)((bits >> 16) & 0x8000);
  uint16_t mant = (uint16_t)((bits >> 13) & 0x03FF);
  int32_t  ex   = (int32_t)((bits >> 23) & 0xFF) - 127 + 15;
  uint16_t h;
  if (ex >= 31)      h = (uint16_t)(sign | 0x7C00);
  else if (ex <= 0)  h = sign;
  else               h = (uint16_t)(sign | ((uint16_t)ex << 10) | mant);
  pOut[row * n_pts + col] = h;
}

// -----------------------------------------------------------------------------
// Stage 2 final: softplus(sigma logit) + sigmoid(rgb logits).
// Runs after layer 3's TCU GEMM.  One task per point.
// Layer 3 Y has shape [MLP_OUT_DIM_PAD x n_points] row-major; row 0 is
// sigma logit, rows 1-3 are RGB logits.
// -----------------------------------------------------------------------------
void kernel_mlp_out_act(kernel_arg_t* __UNIFORM__ arg) {
  auto pY_fp32 = reinterpret_cast<float*>(arg->Y_cur_addr);
  auto pB      = reinterpret_cast<float*>(arg->B_cur_addr);
  auto pSig    = reinterpret_cast<float*>(arg->sigmas_addr);
  auto pRGB    = reinterpret_cast<float*>(arg->rgbs_addr);

  uint32_t tid = blockIdx.x;
  uint32_t n_pts = arg->n_points;

  float s  = pY_fp32[0 * n_pts + tid] + pB[0];
  float r  = pY_fp32[1 * n_pts + tid] + pB[1];
  float g  = pY_fp32[2 * n_pts + tid] + pB[2];
  float b  = pY_fp32[3 * n_pts + tid] + pB[3];
  pSig[tid]        = logf(1.0f + expf(s));
  pRGB[tid*3 + 0]  = 1.0f / (1.0f + expf(-r));
  pRGB[tid*3 + 1]  = 1.0f / (1.0f + expf(-g));
  pRGB[tid*3 + 2]  = 1.0f / (1.0f + expf(-b));
}

// -----------------------------------------------------------------------------
// Stage 3: SIMT alpha compositing (one task per ray).  Same math as the
// SIMT baseline kernel2_body.
// -----------------------------------------------------------------------------
void kernel_composite(kernel_arg_t* __UNIFORM__ arg) {
  auto sigmas = reinterpret_cast<float*>(arg->sigmas_addr);
  auto rgbs   = reinterpret_cast<float*>(arg->rgbs_addr);
  auto deltas = reinterpret_cast<float*>(arg->deltas_addr);
  auto image  = reinterpret_cast<float*>(arg->image_addr);

  uint32_t ray = blockIdx.x;
  uint32_t S   = arg->n_samples;
  uint32_t base = ray * S;

  float T = 1.0f;
  float R = 0.0f, G = 0.0f, B = 0.0f;
  for (uint32_t s = 0; s < S; ++s) {
    float sigma = sigmas[base + s];
    float delta = deltas[base + s];
    float alpha = 1.0f - expf(-sigma * delta);
    float w = alpha * T;
    R += w * rgbs[(base + s) * 3 + 0];
    G += w * rgbs[(base + s) * 3 + 1];
    B += w * rgbs[(base + s) * 3 + 2];
    T *= (1.0f - alpha);
  }
  image[ray * 3 + 0] = R;
  image[ray * 3 + 1] = G;
  image[ray * 3 + 2] = B;
}

// =============================================================================
// Entry point — one spawn per stage, cycles recorded via vx_rdcycle().
// =============================================================================
int main() {
  kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);

  uint64_t t_begin = vx_rdcycle();
  int rc = 0;
  uint32_t n_tasks;

  switch (arg->kernel_id) {
    case KID_RAY_SETUP:
      n_tasks = arg->n_points;  // one task per point for cache-friendly writes
      rc = vx_spawn_threads(1, &n_tasks, nullptr,
                            (vx_kernel_func_cb)kernel_ray_setup, arg);
      break;

    case KID_MLP_GEMM:
      rc = vx_spawn_threads(2, arg->grid_dim, arg->block_dim,
                            (vx_kernel_func_cb)kernel_mlp_gemm, arg);
      break;

    case KID_MLP_ACT:
      n_tasks = arg->N_out_cur * arg->n_points;
      rc = vx_spawn_threads(1, &n_tasks, nullptr,
                            (vx_kernel_func_cb)kernel_mlp_act, arg);
      break;

    case KID_MLP_OUT_ACT:
      n_tasks = arg->n_points;
      rc = vx_spawn_threads(1, &n_tasks, nullptr,
                            (vx_kernel_func_cb)kernel_mlp_out_act, arg);
      break;

    case KID_COMPOSITE:
      n_tasks = arg->n_rays;
      rc = vx_spawn_threads(1, &n_tasks, nullptr,
                            (vx_kernel_func_cb)kernel_composite, arg);
      break;

    default:
      return -1;
  }

  uint64_t t_end = vx_rdcycle();
  arg->kernel_cycles = t_end - t_begin;
  return rc;
}
