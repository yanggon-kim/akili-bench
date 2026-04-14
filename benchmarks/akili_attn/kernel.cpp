#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include "common.h"
#include <cmath>
#include <algorithm>

// =============================================================================
// SIMT attention kernels (KMU-dispatched).
//   qk_body       : S[i,j] = sum_k Q[i,k] * K[k,j]      — one block per (i,j)
//   softmax_body  : P = softmax(S) row-wise             — striped hw-thread pattern
//   pv_body       : O[i,j] = sum_k P[i,k] * V[k,j]      — one block per (i,j)
// =============================================================================

static void qk_body(kernel_arg_t* __UNIFORM__ arg) {
  auto Q = reinterpret_cast<TYPE*>(arg->Q_addr);
  auto K = reinterpret_cast<TYPE*>(arg->K_addr);
  auto S = reinterpret_cast<TYPE*>(arg->S_addr);
  auto N = arg->N;
  auto d = arg->d;

  int col = blockIdx.x;
  int row = blockIdx.y;
  if (row >= (int)N || col >= (int)N) return;

  TYPE sum(0);
  for (uint32_t e = 0; e < d; ++e) {
    sum += Q[row * d + e] * K[e * N + col];
  }
  S[row * N + col] = sum;
}

static void softmax_body(kernel_arg_t* __UNIFORM__ arg) {
  auto S = reinterpret_cast<TYPE*>(arg->S_addr);
  auto P = reinterpret_cast<TYPE*>(arg->P_addr);
  uint32_t N = arg->N;

  uint32_t hw_tid = blockIdx.x * NUM_THREADS + threadIdx.x;
  uint32_t stride = gridDim.x * NUM_THREADS;

  for (uint32_t row = hw_tid; row < N; row += stride) {
    TYPE max_val = S[row * N];
    for (uint32_t col = 1; col < N; ++col) {
      max_val = std::max(max_val, S[row * N + col]);
    }

    TYPE local_P[512];
    TYPE exp_sum = 0;
    for (uint32_t col = 0; col < N; ++col) {
      auto e = std::exp(S[row * N + col] - max_val);
      local_P[col] = e;
      exp_sum += e;
    }
    for (uint32_t col = 0; col < N; ++col) {
      P[row * N + col] = local_P[col] / exp_sum;
    }
  }
}

static void pv_body(kernel_arg_t* __UNIFORM__ arg) {
  auto P = reinterpret_cast<TYPE*>(arg->P_addr);
  auto V = reinterpret_cast<TYPE*>(arg->V_addr);
  auto O = reinterpret_cast<TYPE*>(arg->O_addr);
  auto N = arg->N;
  auto d = arg->d;

  int col = blockIdx.x;
  int row = blockIdx.y;
  if (row >= (int)N || col >= (int)d) return;

  TYPE sum(0);
  for (uint32_t e = 0; e < N; ++e) {
    sum += P[row * N + e] * V[e * d + col];
  }
  O[row * d + col] = sum;
}

// =============================================================================
// Single entry point — host selects the active stage via arg->kernel_id.
// =============================================================================
__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  switch (arg->kernel_id) {
    case KID_QK_SIMT: qk_body(arg);      break;
    case KID_SOFTMAX: softmax_body(arg); break;
    case KID_PV_SIMT: pv_body(arg);      break;
    default: break;
  }
}
