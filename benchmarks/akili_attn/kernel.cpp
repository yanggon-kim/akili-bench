#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include "common.h"
#include <cmath>
#include <algorithm>

// =============================================================================
// SIMT attention kernels.
//   kernel0_body  : S[i,j] = sum_k Q[i,k] * K[k,j]      — one thread per (i,j)
//   kernel1_body  : P = softmax(S) row-wise             — one thread per row
//   kernel2_body  : O[i,j] = sum_k P[i,k] * V[k,j]      — one thread per (i,j)
// =============================================================================

void kernel0_body(kernel_arg_t* __UNIFORM__ arg) {
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

void kernel1_body(kernel_arg_t* __UNIFORM__ arg) {
  auto S = reinterpret_cast<TYPE*>(arg->S_addr);
  auto P = reinterpret_cast<TYPE*>(arg->P_addr);
  auto N = arg->N;

  int row = blockIdx.x;

  TYPE max_val = S[row * N];
  for (uint32_t col = 1; col < N; ++col) {
    max_val = std::max(max_val, S[row * N + col]);
  }

  TYPE local_P[512];
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
  if (row >= (int)N || col >= (int)d) return;

  TYPE sum(0);
  for (uint32_t e = 0; e < N; ++e) {
    sum += P[row * N + e] * V[e * d + col];
  }
  O[row * d + col] = sum;
}

// =============================================================================
// Entry point — wraps each spawn in rdcycle timing.
// =============================================================================
int main() {
  kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);

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
    default:
      return -1;
  }

  uint64_t t_end = vx_rdcycle();
  arg->kernel_cycles = t_end - t_begin;
  return rc;
}
