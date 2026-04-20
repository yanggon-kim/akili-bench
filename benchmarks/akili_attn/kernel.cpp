#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include "common.h"
#include <cmath>
#include <algorithm>


static inline float fast_exp(float x) {
  x = x < -80.0f ? -80.0f : x;
  float u = x * 0.03125f;
  float y = 1.0f + u*(1.0f + u*(0.5f + u*(0.166666667f + u*(0.041666667f + u*0.008333333f))));
  y = y*y; y = y*y; y = y*y; y = y*y; y = y*y;
  return y;
}

// =============================================================================
// akili_attn — SIMT attention (KMU launch API).
//   kernel0_body  : S[i,j] = sum_k Q[i,k] * K[k,j]      — 2D grid (col, row)
//   kernel1_body  : P = softmax(S) row-wise             — 1D grid (row)
//   kernel2_body  : O[i,j] = sum_k P[i,k] * V[k,j]      — 2D grid (col, row)
// =============================================================================

static inline void kernel0_body(kernel_arg_t* arg) {
  auto Q = reinterpret_cast<TYPE*>(arg->Q_addr);
  auto K = reinterpret_cast<TYPE*>(arg->K_addr);
  auto S = reinterpret_cast<TYPE*>(arg->S_addr);
  auto N = arg->N;
  auto d = arg->d;

  uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= N || col >= N) return;

  TYPE sum(0);
  for (uint32_t e = 0; e < d; ++e) {
    sum += Q[row * d + e] * K[e * N + col];
  }
  S[row * N + col] = sum;
}

static inline void kernel1_body(kernel_arg_t* arg) {
  auto S = reinterpret_cast<TYPE*>(arg->S_addr);
  auto P = reinterpret_cast<TYPE*>(arg->P_addr);
  uint32_t N = arg->N;

  uint32_t hw_tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t stride = gridDim.x * blockDim.x;
  for (uint32_t row = hw_tid; row < N; row += stride) {
    TYPE max_val = S[row * N];
    for (uint32_t col = 1; col < N; ++col) {
      max_val = std::max(max_val, S[row * N + col]);
    }

    TYPE local_P[512];
    TYPE exp_sum = 0;
    for (uint32_t col = 0; col < N; ++col) {
      auto e = fast_exp(S[row * N + col] - max_val);
      local_P[col] = e;
      exp_sum += e;
    }
    for (uint32_t col = 0; col < N; ++col) {
      P[row * N + col] = local_P[col] / exp_sum;
    }
  }
}

static inline void kernel2_body(kernel_arg_t* arg) {
  auto P = reinterpret_cast<TYPE*>(arg->P_addr);
  auto V = reinterpret_cast<TYPE*>(arg->V_addr);
  auto O = reinterpret_cast<TYPE*>(arg->O_addr);
  auto N = arg->N;
  auto d = arg->d;

  uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= N || col >= d) return;

  TYPE sum(0);
  for (uint32_t e = 0; e < N; ++e) {
    sum += P[row * N + e] * V[e * d + col];
  }
  O[row * d + col] = sum;
}

// =============================================================================
__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  __rdcycle_time t0 = vx_rdcycle_sync_begin();

  switch (arg->kernel_id) {
    case KID_QK_SIMT: kernel0_body(arg); break;
    case KID_SOFTMAX: kernel1_body(arg); break;
    case KID_PV_SIMT: kernel2_body(arg); break;
    default: break;
  }

  __rdcycle_time t1 = vx_rdcycle_sync_end();
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    auto pCycles = reinterpret_cast<uint32_t*>(arg->cycles_addr);
    uint32_t block_id = blockIdx.y * gridDim.x + blockIdx.x;
    pCycles[block_id] = (uint32_t)vx_rdcycle_sync_diff(t0, t1);
  }
}
