#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include <math.h>
#include "common.h"

// =============================================================================
// akili_NeRF — SIMT full-NeRF forward pass.
//   kernel0_body : ray-AABB slab test + stratified sampling      (non-GEMM)
//   kernel1_body : tiny NeRF MLP forward per (ray, sample) point (GEMM)
//   kernel2_body : per-ray alpha compositing                      (non-GEMM)
// =============================================================================

// ---- Stage 1: ray setup ----
void kernel0_body(kernel_arg_t* __UNIFORM__ arg) {
  auto rays_o  = reinterpret_cast<float*>(arg->rays_o_addr);
  auto rays_d  = reinterpret_cast<float*>(arg->rays_d_addr);
  auto aabb    = reinterpret_cast<float*>(arg->aabb_addr);
  auto pos_out = reinterpret_cast<float*>(arg->sample_pos_addr);
  auto dlt_out = reinterpret_cast<float*>(arg->deltas_addr);

  uint32_t ray = blockIdx.x;
  uint32_t S   = arg->n_samples;
  float min_near = arg->min_near;

  float ox = rays_o[ray*3+0], oy = rays_o[ray*3+1], oz = rays_o[ray*3+2];
  float dx = rays_d[ray*3+0], dy = rays_d[ray*3+1], dz = rays_d[ray*3+2];

  // Slab test against a single AABB [aabb[0..2], aabb[3..5]].
  // Host guarantees all direction components are non-zero.
  float inv_dx = 1.0f / dx;
  float inv_dy = 1.0f / dy;
  float inv_dz = 1.0f / dz;
  float tx1 = (aabb[0] - ox) * inv_dx;
  float tx2 = (aabb[3] - ox) * inv_dx;
  float ty1 = (aabb[1] - oy) * inv_dy;
  float ty2 = (aabb[4] - oy) * inv_dy;
  float tz1 = (aabb[2] - oz) * inv_dz;
  float tz2 = (aabb[5] - oz) * inv_dz;
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

  // Rays that miss collapse all samples to t=tmin with step=0 → alpha=0.
  float step = 0.0f;
  if (tmax > tmin) {
    step = (tmax - tmin) / (float)S;
  } else {
    tmax = tmin;
  }

  for (uint32_t s = 0; s < S; ++s) {
    float t = tmin + ((float)s + 0.5f) * step;
    uint32_t off = (ray * S + s) * 3;
    pos_out[off + 0] = ox + t * dx;
    pos_out[off + 1] = oy + t * dy;
    pos_out[off + 2] = oz + t * dz;
    dlt_out[ray * S + s] = step;
  }
}

// ---- Stage 2: tiny NeRF MLP forward ----
void kernel1_body(kernel_arg_t* __UNIFORM__ arg) {
  auto pos    = reinterpret_cast<float*>(arg->sample_pos_addr);
  auto W0     = reinterpret_cast<float*>(arg->w0_addr);
  auto B0     = reinterpret_cast<float*>(arg->b0_addr);
  auto W1     = reinterpret_cast<float*>(arg->w1_addr);
  auto B1     = reinterpret_cast<float*>(arg->b1_addr);
  auto W2     = reinterpret_cast<float*>(arg->w2_addr);
  auto B2     = reinterpret_cast<float*>(arg->b2_addr);
  auto W3     = reinterpret_cast<float*>(arg->w3_addr);
  auto B3     = reinterpret_cast<float*>(arg->b3_addr);
  auto sigmas = reinterpret_cast<float*>(arg->sigmas_addr);
  auto rgbs   = reinterpret_cast<float*>(arg->rgbs_addr);

  uint32_t tid = blockIdx.x;
  float x = pos[tid * 3 + 0];
  float y = pos[tid * 3 + 1];
  float z = pos[tid * 3 + 2];

  // Positional encoding → 27 features packed into h_a[0..26].
  // Layout: [x, y, z, sin(f0 x), cos(f0 x), sin(f0 y), cos(f0 y),
  //          sin(f0 z), cos(f0 z), sin(f1 x), cos(f1 x), ...]
  // Slots 27..31 stay 0; they are never read by layer 0 (k < MLP_IN_DIM).
  float h_a[MLP_W];
  float h_b[MLP_W];
  for (int i = 0; i < MLP_W; ++i) h_a[i] = 0.0f;
  h_a[0] = x;
  h_a[1] = y;
  h_a[2] = z;
  float freq = 1.0f;
  for (int L = 0; L < PE_L; ++L) {
    h_a[3 + L*6 + 0] = sinf(freq * x);
    h_a[3 + L*6 + 1] = cosf(freq * x);
    h_a[3 + L*6 + 2] = sinf(freq * y);
    h_a[3 + L*6 + 3] = cosf(freq * y);
    h_a[3 + L*6 + 4] = sinf(freq * z);
    h_a[3 + L*6 + 5] = cosf(freq * z);
    freq *= 2.0f;
  }

  // Layer 0: 27 -> 32 + ReLU  (h_a -> h_b)
  for (int j = 0; j < MLP_W; ++j) {
    float sum = B0[j];
    for (int k = 0; k < MLP_IN_DIM; ++k) sum += h_a[k] * W0[k * MLP_W + j];
    h_b[j] = (sum > 0.0f) ? sum : 0.0f;
  }
  // Layer 1: 32 -> 32 + ReLU  (h_b -> h_a)
  for (int j = 0; j < MLP_W; ++j) {
    float sum = B1[j];
    for (int k = 0; k < MLP_W; ++k) sum += h_b[k] * W1[k * MLP_W + j];
    h_a[j] = (sum > 0.0f) ? sum : 0.0f;
  }
  // Layer 2: 32 -> 32 + ReLU  (h_a -> h_b)
  for (int j = 0; j < MLP_W; ++j) {
    float sum = B2[j];
    for (int k = 0; k < MLP_W; ++k) sum += h_a[k] * W2[k * MLP_W + j];
    h_b[j] = (sum > 0.0f) ? sum : 0.0f;
  }
  // Layer 3: 32 -> 4 (linear, no activation)
  float o0 = B3[0];
  float o1 = B3[1];
  float o2 = B3[2];
  float o3 = B3[3];
  for (int k = 0; k < MLP_W; ++k) {
    float hv = h_b[k];
    o0 += hv * W3[k * MLP_OUT_DIM + 0];
    o1 += hv * W3[k * MLP_OUT_DIM + 1];
    o2 += hv * W3[k * MLP_OUT_DIM + 2];
    o3 += hv * W3[k * MLP_OUT_DIM + 3];
  }

  // NeRF activation convention: sigma via softplus, rgb via sigmoid.
  sigmas[tid]             = logf(1.0f + expf(o0));
  rgbs[tid * 3 + 0]       = 1.0f / (1.0f + expf(-o1));
  rgbs[tid * 3 + 1]       = 1.0f / (1.0f + expf(-o2));
  rgbs[tid * 3 + 2]       = 1.0f / (1.0f + expf(-o3));
}

// ---- Stage 3: alpha compositing ----
void kernel2_body(kernel_arg_t* __UNIFORM__ arg) {
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
      n_tasks = arg->n_rays;
      rc = vx_spawn_threads(1, &n_tasks, nullptr,
                            (vx_kernel_func_cb)kernel0_body, arg);
      break;
    case KID_MLP_FWD:
      n_tasks = arg->n_rays * arg->n_samples;
      rc = vx_spawn_threads(1, &n_tasks, nullptr,
                            (vx_kernel_func_cb)kernel1_body, arg);
      break;
    case KID_COMPOSITE:
      n_tasks = arg->n_rays;
      rc = vx_spawn_threads(1, &n_tasks, nullptr,
                            (vx_kernel_func_cb)kernel2_body, arg);
      break;
    default:
      return -1;
  }

  uint64_t t_end = vx_rdcycle();
  arg->kernel_cycles = t_end - t_begin;
  return rc;
}
