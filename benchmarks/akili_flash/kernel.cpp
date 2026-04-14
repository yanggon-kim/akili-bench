#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_print.h>
#include <cmath>
#include <cstring>
#include "common.h"

// =============================================================================
// akili_flash — SIMT FlashAttention (KMU-dispatched).
//
// Two code paths live here:
//
//   (A) Fused online-softmax kernel (template flash_kernel_body<HEAD_DIM,
//       BLOCK_SIZE_C>). One launch; each thread walks one Q row and streams
//       K/V blocks to build the output incrementally. Active only when
//       d ∈ {1,2,4,8,16}. Host picks the (HEAD_DIM, BLOCK_SIZE_C) specialization.
//
//   (B) Unfused 3-stage attention kernels (flash_qk_simt → flash_softmax_body
//       → flash_pv_simt). Used when d falls outside the fused template's set.
// =============================================================================

template<uint32_t HEAD_DIM, uint32_t BLOCK_SIZE_C>
static void flash_kernel_body(kernel_arg_t *arg) {
  float* Q_ptr = reinterpret_cast<float*>(arg->Q_addr);
  float* K_ptr = reinterpret_cast<float*>(arg->K_addr);
  float* V_ptr = reinterpret_cast<float*>(arg->V_addr);
  float* O_ptr = reinterpret_cast<float*>(arg->O_addr);

  auto seq_len = arg->seq_len;
  auto block_size_r = arg->block_size_r;

  auto local_ptr = __local_mem((block_size_r + 2 * BLOCK_SIZE_C) * HEAD_DIM * sizeof(float));
  auto local_Q = (float*)local_ptr;
  auto local_K = (float*)local_Q + block_size_r * HEAD_DIM;
  auto local_V = (float*)local_K + BLOCK_SIZE_C * HEAD_DIM;

  auto g_row = blockIdx.x * blockDim.x + threadIdx.x;
  auto l_row = threadIdx.x;
  auto g_row_offset = g_row * HEAD_DIM;
  auto l_row_offset = l_row * HEAD_DIM;

  for (uint32_t k = 0; k < HEAD_DIM; ++k)
    local_Q[l_row_offset + k] = Q_ptr[g_row_offset + k];
  float* Q_row = local_Q + l_row * HEAD_DIM;

  float O_buf[HEAD_DIM];
  for (uint32_t k = 0; k < HEAD_DIM; ++k) O_buf[k] = 0.0f;

  float sp_buf[BLOCK_SIZE_C];

  float m = -INFINITY;
  float l = 0.0f;

  for (uint32_t j = 0; j < seq_len; j += BLOCK_SIZE_C) {
    auto block_offset = j * HEAD_DIM;

    for (uint32_t k = 0; k < BLOCK_SIZE_C / block_size_r; ++k) {
      auto row = k * block_size_r + l_row;
      auto row_offset = row * HEAD_DIM;
      for (uint32_t col = 0; col < HEAD_DIM; ++col) {
        auto offset = row_offset + col;
        local_K[offset] = K_ptr[block_offset + offset];
        local_V[col * BLOCK_SIZE_C + row] = V_ptr[block_offset + offset];
      }
    }
    __syncthreads();

    for (uint32_t k = 0; k < BLOCK_SIZE_C; ++k) sp_buf[k] = 0;
    for (uint32_t k = 0; k < BLOCK_SIZE_C; ++k)
      for (uint32_t elem = 0; elem < HEAD_DIM; ++elem)
        sp_buf[k] += Q_row[elem] * local_K[k * HEAD_DIM + elem];

    float rowmax = sp_buf[0];
    for (uint32_t k = 1; k < BLOCK_SIZE_C; ++k)
      rowmax = (sp_buf[k] > rowmax ? sp_buf[k] : rowmax);

    for (uint32_t k = 0; k < BLOCK_SIZE_C; ++k)
      sp_buf[k] = expf(sp_buf[k] - rowmax);

    float rowsum = 0.0f;
    for (uint32_t k = 0; k < BLOCK_SIZE_C; ++k) rowsum += sp_buf[k];

    float new_m = (m > rowmax ? m : rowmax);
    float new_l = expf(m - new_m) * l + expf(rowmax - new_m) * rowsum;
    float old_weight = expf(m - new_m);
    float new_weight = expf(rowmax - new_m);

    for (uint32_t k = 0; k < HEAD_DIM; ++k) {
      float dot = 0.0f;
      for (uint32_t elem = 0; elem < BLOCK_SIZE_C; ++elem)
        dot += sp_buf[elem] * local_V[k * BLOCK_SIZE_C + elem];
      O_buf[k] = old_weight * O_buf[k] + new_weight * dot;
    }

    m = new_m;
    l = new_l;
    __syncthreads();
  }

  float inv_l = 1.0f / l;
  for (uint32_t k = 0; k < HEAD_DIM; ++k)
    O_ptr[g_row_offset + k] = O_buf[k] * inv_l;
}

// =============================================================================
// Unfused 3-stage SIMT kernels.
// =============================================================================
static void flash_qk_simt(kernel_arg_t* __UNIFORM__ arg) {
  auto Q = reinterpret_cast<float*>(arg->Q_addr);
  auto K = reinterpret_cast<float*>(arg->K_addr);
  auto S = reinterpret_cast<float*>(arg->S_addr);
  uint32_t N = arg->N;
  uint32_t d = arg->d;
  int col = blockIdx.x;
  int row = blockIdx.y;
  if (row < (int)N && col < (int)N) {
    float sum = 0;
    for (uint32_t e = 0; e < d; ++e) sum += Q[row * d + e] * K[e * N + col];
    S[row * N + col] = sum;
  }
}

static void flash_softmax_body(kernel_arg_t* __UNIFORM__ arg) {
  auto S = reinterpret_cast<float*>(arg->S_addr);
  auto P = reinterpret_cast<float*>(arg->P_addr);
  uint32_t N = arg->N;

  uint32_t hw_tid = blockIdx.x * NUM_THREADS + threadIdx.x;
  uint32_t stride = gridDim.x * NUM_THREADS;

  for (uint32_t row = hw_tid; row < N; row += stride) {
    float max_val = S[row * N];
    for (uint32_t col = 1; col < N; ++col) {
      float v = S[row * N + col];
      if (v > max_val) max_val = v;
    }
    float local_P[512];
    float exp_sum = 0;
    for (uint32_t col = 0; col < N; ++col) {
      float e = expf(S[row * N + col] - max_val);
      local_P[col] = e;
      exp_sum += e;
    }
    for (uint32_t col = 0; col < N; ++col) P[row * N + col] = local_P[col] / exp_sum;
  }
}

static void flash_pv_simt(kernel_arg_t* __UNIFORM__ arg) {
  auto P = reinterpret_cast<float*>(arg->P_addr);
  auto V = reinterpret_cast<float*>(arg->V_addr);
  auto O = reinterpret_cast<float*>(arg->O_addr);
  uint32_t N = arg->N;
  uint32_t d = arg->d;
  int col = blockIdx.x;
  int row = blockIdx.y;
  if (row < (int)N && col < (int)d) {
    float sum = 0;
    for (uint32_t e = 0; e < N; ++e) sum += P[row * N + e] * V[e * d + col];
    O[row * d + col] = sum;
  }
}

// Fused entry — host sets head_dim/block_size_c and we dispatch to the
// corresponding template specialization.
static void flash_kernel_entry(kernel_arg_t* arg) {
  switch (arg->head_dim) {
    case 1:
      switch (arg->block_size_c) {
        case 8:   flash_kernel_body<1,8>(arg);   return;
        case 16:  flash_kernel_body<1,16>(arg);  return;
        case 32:  flash_kernel_body<1,32>(arg);  return;
        case 64:  flash_kernel_body<1,64>(arg);  return;
        case 128: flash_kernel_body<1,128>(arg); return;
      }
      break;
    case 2:
      switch (arg->block_size_c) {
        case 8:  flash_kernel_body<2,8>(arg);  return;
        case 16: flash_kernel_body<2,16>(arg); return;
        case 32: flash_kernel_body<2,32>(arg); return;
        case 64: flash_kernel_body<2,64>(arg); return;
      }
      break;
    case 4:
      switch (arg->block_size_c) {
        case 8:  flash_kernel_body<4,8>(arg);  return;
        case 16: flash_kernel_body<4,16>(arg); return;
        case 32: flash_kernel_body<4,32>(arg); return;
      }
      break;
    case 8:
      switch (arg->block_size_c) {
        case 8:  flash_kernel_body<8,8>(arg);  return;
        case 16: flash_kernel_body<8,16>(arg); return;
      }
      break;
    case 16:
      switch (arg->block_size_c) {
        case 8: flash_kernel_body<16,8>(arg); return;
      }
      break;
  }
}

// =============================================================================
// Single entry point — host selects the active stage via arg->kernel_id.
// =============================================================================
__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  switch (arg->kernel_id) {
    case KID_FLASH_FUSED: flash_kernel_entry(arg);  break;
    case KID_QK_SIMT:     flash_qk_simt(arg);       break;
    case KID_SOFTMAX:     flash_softmax_body(arg);  break;
    case KID_PV_SIMT:     flash_pv_simt(arg);       break;
    default: break;
  }
}
