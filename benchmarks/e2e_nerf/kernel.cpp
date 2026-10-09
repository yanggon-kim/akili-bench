#include "e2e_linear.h"
#include "common.h"
#include <math.h>

using namespace e2e;

__kernel void linear(linear_args_t* __UNIFORM__ arg) { linear_body(arg); }

// softplus(x) = ln(1 + e^x) via the shared exp2 and a float log2.
static inline float softplus_dev(float x) {
  const float y = 1.0f + exp2_dev(1.44269504f * x);
  const uint32_t b = f2u(y);
  const float m = u2f((b & 0x7fffffu) | 0x3f800000u);      // [1, 2)
  const float e = (float)((int32_t)(b >> 23) - 127);
  const float t = (m - 1.0f) / (m + 1.0f), t2 = t * t;    // ln(m) = 2 atanh(t)
  return 0.693147182f * e + 2.0f * t * (1.0f + t2 * (0.333333343f + t2 * (0.2f + t2 * 0.142857149f)));
}

// Positional-encoding feature f of a 3-vector with nf frequencies; 0 past the end.
static inline float pe_feature(const float* v, uint32_t f, uint32_t nf) {
  if (f < 3) return v[f];
  f -= 3;
  if (f >= 6 * nf) return 0.0f;
  const uint32_t l = f / 6, rem = f % 6, axis = rem / 2;
  const float x = v[axis] * (float)(1u << l);
  return (rem & 1) ? cos_poly(x) : sin_poly(x);
}

// One thread per (point, run of QRUN encoded features): ray-AABB slab test,
// stratified sample position, then quantized PE(xyz) into x0 and the skip
// half of x5, and PE(dir) into x8.
__kernel void ray_setup(nerf_args_t* __UNIFORM__ arg) {
  const float* ro = reinterpret_cast<const float*>(arg->rays_o);
  const float* rd = reinterpret_cast<const float*>(arg->rays_d);
  const uint32_t S = arg->n_samples, P = arg->n_points;
  const uint32_t rx = arg->k0 / QRUN, rdn = arg->kd / QRUN, runs = rx + rdn;
  const uint32_t nkt0 = arg->k0 / KT, nkt5 = (NERF_W + arg->k0) / KT, nkt8 = (NERF_W + arg->kd) / KT;
  for (uint32_t g = blockIdx.x * blockDim.x + threadIdx.x; g < P * runs; g += gridDim.x * blockDim.x) {
    const uint32_t pt = g / runs, run = g % runs, ray = pt / S, s = pt % S;
    float o[3] = {ro[ray * 3], ro[ray * 3 + 1], ro[ray * 3 + 2]};
    float d[3] = {rd[ray * 3], rd[ray * 3 + 1], rd[ray * 3 + 2]};
    float tmin = arg->near, tmax = arg->far;
    for (uint32_t a = 0; a < 3; ++a) {
      float inv = 1.0f / d[a];
      float t1 = (-1.0f - o[a]) * inv, t2 = (1.0f - o[a]) * inv;
      float lo = t1 < t2 ? t1 : t2, hi = t1 < t2 ? t2 : t1;
      tmin = lo > tmin ? lo : tmin;
      tmax = hi < tmax ? hi : tmax;
    }
    const float step = tmax > tmin ? (tmax - tmin) / (float)S : 0.0f;
    const float t = tmin + ((float)s + 0.5f) * step;
    float p[3] = {o[0] + t * d[0], o[1] + t * d[1], o[2] + t * d[2]};
    if (run == 0) reinterpret_cast<float*>(arg->deltas)[pt] = step;
    // Both quantizer calls stay convergent (MX block amax is a lane shuffle):
    // dir runs have no x5 half and repeat their x8 store instead.
    const bool isx = run < rx;
    const uint32_t c0 = (isx ? run : run - rx) * QRUN;
    float v[QRUN];
    for (uint32_t i = 0; i < QRUN; ++i) v[i] = isx ? pe_feature(p, c0 + i, NERF_PE_X) : pe_feature(d, c0 + i, NERF_PE_D);
    uint32_t* w8 = reinterpret_cast<uint32_t*>(arg->x8) + blk_word(pt, (NERF_W + c0) / F::epw, LIN_BM, nkt8);
    uint8_t* s8 = IS_MX ? reinterpret_cast<uint8_t*>(arg->x8_sc) + blk_sc(pt, NERF_W + c0, LIN_BM, nkt8) : nullptr;
    uint32_t* wa = isx ? reinterpret_cast<uint32_t*>(arg->x0) + blk_word(pt, c0 / F::epw, LIN_BM, nkt0) : w8;
    uint8_t* sa = isx ? (IS_MX ? reinterpret_cast<uint8_t*>(arg->x0_sc) + blk_sc(pt, c0, LIN_BM, nkt0) : nullptr) : s8;
    uint32_t* wb = isx ? reinterpret_cast<uint32_t*>(arg->x5) + blk_word(pt, (NERF_W + c0) / F::epw, LIN_BM, nkt5) : w8;
    uint8_t* sb = isx ? (IS_MX ? reinterpret_cast<uint8_t*>(arg->x5_sc) + blk_sc(pt, NERF_W + c0, LIN_BM, nkt5) : nullptr) : s8;
    quant_store_run(v, QRUN, isx ? arg->ts_x0 : arg->ts_x8, wa, sa);
    quant_store_run(v, QRUN, isx ? arg->ts_x5 : arg->ts_x8, wb, sb);
  }
}

// One thread per sample point: rgb head (3x128 dot products), softplus
// density -> alpha, sigmoid color.
__kernel void point_head(nerf_args_t* __UNIFORM__ arg) {
  const float* h8 = reinterpret_cast<const float*>(arg->h8);
  const float* w9 = reinterpret_cast<const float*>(arg->w9);
  const float* b9 = reinterpret_cast<const float*>(arg->b9);
  const float* sig = reinterpret_cast<const float*>(arg->sigma_raw);
  const float* dl = reinterpret_cast<const float*>(arg->deltas);
  float* rgba = reinterpret_cast<float*>(arg->rgba);
  for (uint32_t pt = blockIdx.x * blockDim.x + threadIdx.x; pt < arg->n_points; pt += gridDim.x * blockDim.x) {
    const float* h = h8 + pt * NERF_WV;
    float l0 = b9[0], l1 = b9[1], l2 = b9[2];
    for (uint32_t k = 0; k < NERF_WV; ++k) {
      const float hk = h[k];
      l0 = fma_dev(hk, w9[k], l0);
      l1 = fma_dev(hk, w9[NERF_WV + k], l1);
      l2 = fma_dev(hk, w9[2 * NERF_WV + k], l2);
    }
    const float sr = sig[pt];
    const float sigma = sr > 15.0f ? sr : softplus_dev(sr);
    rgba[pt * 4 + 0] = 1.0f / (1.0f + exp2_dev(-1.44269504f * l0));
    rgba[pt * 4 + 1] = 1.0f / (1.0f + exp2_dev(-1.44269504f * l1));
    rgba[pt * 4 + 2] = 1.0f / (1.0f + exp2_dev(-1.44269504f * l2));
    rgba[pt * 4 + 3] = 1.0f - exp2_dev(-1.44269504f * sigma * dl[pt]);
  }
}

// One thread per ray: front-to-back alpha compositing.
__kernel void composite(nerf_args_t* __UNIFORM__ arg) {
  const float* rgba = reinterpret_cast<const float*>(arg->rgba);
  float* img = reinterpret_cast<float*>(arg->image);
  const uint32_t S = arg->n_samples;
  for (uint32_t ray = blockIdx.x * blockDim.x + threadIdx.x; ray < arg->n_rays; ray += gridDim.x * blockDim.x) {
    float T = 1.0f, c0 = 0.0f, c1 = 0.0f, c2 = 0.0f;
    for (uint32_t s = 0; s < S; ++s) {
      const float* p = rgba + (ray * S + s) * 4;
      const float w = p[3] * T;
      c0 += w * p[0];
      c1 += w * p[1];
      c2 += w * p[2];
      T *= 1.0f - p[3];
    }
    img[ray * 3] = c0;
    img[ray * 3 + 1] = c1;
    img[ray * 3 + 2] = c2;
  }
}
