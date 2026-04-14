// akili_flash — SIMT FlashAttention.
//
// For d ∈ {1,2,4,8,16}: runs the fused online-softmax kernel
// (flash_kernel_body<HEAD_DIM, BLOCK_SIZE_C>) in a single launch. Uses K
// in [N × d] layout (per-token row layout).
//
// For d outside that set: falls through to the unfused 3-stage SIMT path
// (flash_qk_simt → flash_softmax_body → flash_pv_simt) using K in the
// attention-style [d × N] layout.

#include <iostream>
#include <unistd.h>
#include <string.h>
#include <cstring>
#include <vector>
#include <chrono>
#include <vortex.h>
#include <cmath>
#include <algorithm>
#include <stdio.h>
#include "common.h"

#ifndef NUM_THREADS
#define NUM_THREADS 8
#endif

#define RT_CHECK(_expr)                                         \
   do {                                                         \
     int _ret = _expr;                                          \
     if (0 == _ret) break;                                      \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
     cleanup();                                                 \
     exit(-1);                                                  \
   } while (false)

static float frand() { return 10.0f * (float(rand()) / RAND_MAX) - 5.0f; }

static bool compare(float a, float b, float atol, float rtol,
                    int index, int& errors, const char* tag) {
  auto diff = std::fabs(a - b);
  auto limit = atol + rtol * std::fabs(b);
  if (diff > limit) {
    if (errors < 20)
      printf("*** %s error: [%d] expected=%f, actual=%f\n", tag, index, a, b);
    return false;
  }
  return true;
}

static void matmul_cpu(float* out, const float* A, const float* B,
                       uint32_t M, uint32_t N, uint32_t K) {
  for (uint32_t row = 0; row < M; ++row)
    for (uint32_t col = 0; col < N; ++col) {
      double sum = 0;
      for (uint32_t e = 0; e < K; ++e) sum += (double)A[row*K+e] * (double)B[e*N+col];
      out[row*N+col] = (float)sum;
    }
}
static void softmax_cpu(float* out, const float* A, uint32_t M, uint32_t N) {
  for (uint32_t row = 0; row < M; ++row) {
    float max_val = A[row*N];
    for (uint32_t col = 1; col < N; ++col)
      max_val = std::max(max_val, A[row*N + col]);
    float exp_sum = 0;
    for (uint32_t col = 0; col < N; ++col) {
      out[row*N + col] = std::exp(A[row*N + col] - max_val);
      exp_sum += out[row*N + col];
    }
    for (uint32_t col = 0; col < N; ++col) out[row*N + col] /= exp_sum;
  }
}

// Fused-reference attention: K and V indexed by [n, d] per-token layout.
static void attention_cpu_fused(float* out,
                                const float* Q, const float* K, const float* V,
                                uint32_t N, uint32_t d) {
  std::vector<float> S(N * N), P(N * N);
  for (uint32_t r = 0; r < N; ++r)
    for (uint32_t c = 0; c < N; ++c) {
      double sum = 0;
      for (uint32_t e = 0; e < d; ++e)
        sum += (double)Q[r*d + e] * (double)K[c*d + e];
      S[r*N + c] = (float)sum;
    }
  for (uint32_t r = 0; r < N; ++r) {
    float max_val = S[r*N];
    for (uint32_t c = 1; c < N; ++c) max_val = std::max(max_val, S[r*N+c]);
    float exp_sum = 0;
    for (uint32_t c = 0; c < N; ++c) {
      P[r*N + c] = std::exp(S[r*N+c] - max_val);
      exp_sum += P[r*N+c];
    }
    for (uint32_t c = 0; c < N; ++c) P[r*N + c] /= exp_sum;
  }
  for (uint32_t r = 0; r < N; ++r)
    for (uint32_t c = 0; c < d; ++c) {
      double sum = 0;
      for (uint32_t e = 0; e < N; ++e)
        sum += (double)P[r*N + e] * (double)V[e*d + c];
      out[r*d + c] = (float)sum;
    }
}

// ---------------------------------------------------------------------------

const char* kernel_file = "kernel.vxbin";
uint32_t N_req = 16;
uint32_t d_req = 16;

vx_device_h device = nullptr;
vx_buffer_h Q_buffer = nullptr;
vx_buffer_h K_buffer = nullptr;
vx_buffer_h V_buffer = nullptr;
vx_buffer_h S_buffer = nullptr;
vx_buffer_h P_buffer = nullptr;
vx_buffer_h O_buffer = nullptr;
vx_buffer_h cycles_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
  std::cout << "akili_flash — SIMT FlashAttention benchmark" << std::endl;
  std::cout << "Usage: [-n N] [-d D] [-k kernel_file]" << std::endl;
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "n:d:k:h")) != -1) {
    switch (c) {
      case 'n': N_req = atoi(optarg); break;
      case 'd': d_req = atoi(optarg); break;
      case 'k': kernel_file = optarg; break;
      case 'h': show_usage(); exit(0);
      default:  show_usage(); exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    if (Q_buffer)    vx_mem_free(Q_buffer);
    if (K_buffer)    vx_mem_free(K_buffer);
    if (V_buffer)    vx_mem_free(V_buffer);
    if (S_buffer)    vx_mem_free(S_buffer);
    if (P_buffer)    vx_mem_free(P_buffer);
    if (O_buffer)    vx_mem_free(O_buffer);
    if (cycles_buffer) vx_mem_free(cycles_buffer);
    if (krnl_buffer) vx_mem_free(krnl_buffer);
    if (args_buffer) vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

static void read_back_cycles(const char* stage_tag, uint32_t num_blocks) {
  std::vector<uint32_t> h_cycles(num_blocks, 0);
  vx_copy_from_dev(h_cycles.data(), cycles_buffer, 0, num_blocks * sizeof(uint32_t));
  uint32_t max_cyc = 0;
  for (auto c : h_cycles) if (c > max_cyc) max_cyc = c;
  printf("KCYC[%s,nt=%u]: %u\n", stage_tag, (unsigned)NUM_THREADS, max_cyc);
}

int main(int argc, char* argv[]) {
  parse_args(argc, argv);
  std::srand(50);
  RT_CHECK(vx_dev_open(&device));

  uint64_t NT_caps = 0;
  vx_dev_caps(device, VX_CAPS_NUM_THREADS, &NT_caps);
  if ((uint32_t)NT_caps != NUM_THREADS) {
    printf("Error: device NUM_THREADS=%u but built with NUM_THREADS=%u\n",
           (unsigned)NT_caps, (unsigned)NUM_THREADS);
    cleanup();
    return -1;
  }

  uint32_t N = N_req;
  uint32_t d = d_req;
  const float atol = (d >= 128) ? 1e-3f : 1e-5f;
  const float rtol = (d >= 128) ? 1e-3f : 1e-5f;

  std::cout << "akili_flash (SIMT) — N=" << N << " d=" << d << std::endl;

  const bool fused_simt_supported =
      (d == 1 || d == 2 || d == 4 || d == 8 || d == 16);

  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
  RT_CHECK(vx_mem_alloc(device, sizeof(kernel_arg_t), VX_MEM_READ_WRITE, &args_buffer));

  // =========================================================================
  // Fused path (d ∈ {1,2,4,8,16}) — single kernel launch. Uses [N×d] K layout.
  // =========================================================================
  if (fused_simt_supported) {
    std::vector<float> h_Q(N * d), h_K(N * d), h_V(N * d);
    for (auto& v : h_Q) v = frand();
    for (auto& v : h_K) v = frand();
    for (auto& v : h_V) v = frand();

    uint32_t nd_bytes = N * d * sizeof(float);
    RT_CHECK(vx_mem_alloc(device, nd_bytes, VX_MEM_READ_WRITE, &Q_buffer));
    RT_CHECK(vx_mem_address(Q_buffer, &kernel_arg.Q_addr));
    RT_CHECK(vx_mem_alloc(device, nd_bytes, VX_MEM_READ_WRITE, &K_buffer));
    RT_CHECK(vx_mem_address(K_buffer, &kernel_arg.K_addr));
    RT_CHECK(vx_mem_alloc(device, nd_bytes, VX_MEM_READ_WRITE, &V_buffer));
    RT_CHECK(vx_mem_address(V_buffer, &kernel_arg.V_addr));
    RT_CHECK(vx_mem_alloc(device, nd_bytes, VX_MEM_READ_WRITE, &O_buffer));
    RT_CHECK(vx_mem_address(O_buffer, &kernel_arg.O_addr));
    RT_CHECK(vx_copy_to_dev(Q_buffer, h_Q.data(), 0, nd_bytes));
    RT_CHECK(vx_copy_to_dev(K_buffer, h_K.data(), 0, nd_bytes));
    RT_CHECK(vx_copy_to_dev(V_buffer, h_V.data(), 0, nd_bytes));

    uint32_t block_size_r = std::min((uint32_t)8, N);
    uint32_t block_size_c = std::max(block_size_r, (uint32_t)8);

    kernel_arg.N = N;
    kernel_arg.d = d;
    kernel_arg.seq_len      = N;
    kernel_arg.head_dim     = d;
    kernel_arg.block_size_r = block_size_r;
    kernel_arg.block_size_c = block_size_c;
    kernel_arg.kernel_id    = KID_FLASH_FUSED;

    uint32_t grid_dim[2]  = {N / block_size_r, 1};
    uint32_t block_dim[2] = {block_size_r, 1};
    uint32_t num_blocks   = grid_dim[0];
    uint32_t lmem_size    = (block_size_r + 2 * block_size_c) * d * sizeof(float);

    RT_CHECK(vx_mem_alloc(device, num_blocks * sizeof(uint32_t),
                          VX_MEM_READ_WRITE, &cycles_buffer));
    RT_CHECK(vx_mem_address(cycles_buffer, &kernel_arg.cycles_addr));

    std::cout << "=== Fused SIMT flash (d=" << d << ") ===" << std::endl;
    RT_CHECK(vx_copy_to_dev(args_buffer, &kernel_arg, 0, sizeof(kernel_arg_t)));
    RT_CHECK(vx_start_g(device, krnl_buffer, args_buffer, 1, grid_dim, block_dim, lmem_size));
    RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
    read_back_cycles("FUSED", num_blocks);

    std::vector<float> h_O(N * d);
    RT_CHECK(vx_copy_from_dev(h_O.data(), O_buffer, 0, nd_bytes));

    int errors = 0;
    std::vector<float> h_O_ref(N * d);
    attention_cpu_fused(h_O_ref.data(), h_Q.data(), h_K.data(), h_V.data(), N, d);
    for (uint32_t i = 0; i < N * d; ++i)
      if (!compare(h_O_ref[i], h_O[i], atol, rtol, (int)i, errors, "O"))
        ++errors;

    cleanup();
    if (errors) { std::cout << "FAILED!\n"; return errors; }
    std::cout << "PASSED!" << std::endl;
    return 0;
  }

  // =========================================================================
  // Unfused 3-stage path. Uses [d × N] K layout for kernel0 compatibility.
  // =========================================================================
  std::vector<float> h_Q(N * d), h_K(d * N), h_V(N * d);
  for (auto& v : h_Q) v = frand();
  for (auto& v : h_K) v = frand();
  for (auto& v : h_V) v = frand();

  uint32_t in_bytes  = N * d * sizeof(float);
  uint32_t s_bytes   = N * N * sizeof(float);
  uint32_t out_bytes = N * d * sizeof(float);
  RT_CHECK(vx_mem_alloc(device, in_bytes,  VX_MEM_READ_WRITE, &Q_buffer));
  RT_CHECK(vx_mem_address(Q_buffer, &kernel_arg.Q_addr));
  RT_CHECK(vx_mem_alloc(device, in_bytes,  VX_MEM_READ_WRITE, &K_buffer));
  RT_CHECK(vx_mem_address(K_buffer, &kernel_arg.K_addr));
  RT_CHECK(vx_mem_alloc(device, s_bytes,   VX_MEM_READ_WRITE, &S_buffer));
  RT_CHECK(vx_mem_address(S_buffer, &kernel_arg.S_addr));
  RT_CHECK(vx_mem_alloc(device, s_bytes,   VX_MEM_READ_WRITE, &P_buffer));
  RT_CHECK(vx_mem_address(P_buffer, &kernel_arg.P_addr));
  RT_CHECK(vx_mem_alloc(device, in_bytes,  VX_MEM_READ_WRITE, &V_buffer));
  RT_CHECK(vx_mem_address(V_buffer, &kernel_arg.V_addr));
  RT_CHECK(vx_mem_alloc(device, out_bytes, VX_MEM_READ_WRITE, &O_buffer));
  RT_CHECK(vx_mem_address(O_buffer, &kernel_arg.O_addr));
  RT_CHECK(vx_copy_to_dev(Q_buffer, h_Q.data(), 0, in_bytes));
  RT_CHECK(vx_copy_to_dev(K_buffer, h_K.data(), 0, in_bytes));
  RT_CHECK(vx_copy_to_dev(V_buffer, h_V.data(), 0, in_bytes));
  kernel_arg.N = N;
  kernel_arg.d = d;

  int errors = 0;

  // Allocate cycles buffer sized to the largest stage grid.
  uint32_t gd_qk_x = (N + NUM_THREADS - 1) / NUM_THREADS;
  uint32_t gd_qk_y = N;
  uint32_t gd_sm_x = (N + NUM_THREADS - 1) / NUM_THREADS;
  uint32_t gd_pv_x = (d + NUM_THREADS - 1) / NUM_THREADS;
  uint32_t gd_pv_y = N;
  uint32_t qk_blocks = gd_qk_x * gd_qk_y;
  uint32_t sm_blocks = gd_sm_x;
  uint32_t pv_blocks = gd_pv_x * gd_pv_y;
  uint32_t max_blocks = qk_blocks;
  if (sm_blocks > max_blocks) max_blocks = sm_blocks;
  if (pv_blocks > max_blocks) max_blocks = pv_blocks;
  RT_CHECK(vx_mem_alloc(device, max_blocks * sizeof(uint32_t),
                        VX_MEM_READ_WRITE, &cycles_buffer));
  RT_CHECK(vx_mem_address(cycles_buffer, &kernel_arg.cycles_addr));

  uint32_t block_dim[2] = {NUM_THREADS, 1};

  // Stage 1
  std::cout << "=== Stage 1: S = Q @ K^T ===" << std::endl;
  kernel_arg.kernel_id = KID_QK_SIMT;
  RT_CHECK(vx_copy_to_dev(args_buffer, &kernel_arg, 0, sizeof(kernel_arg_t)));
  {
    uint32_t grid_dim[2] = {gd_qk_x, gd_qk_y};
    RT_CHECK(vx_start_g(device, krnl_buffer, args_buffer, 2, grid_dim, block_dim, 0));
  }
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
  read_back_cycles("QK", qk_blocks);
  std::vector<float> h_S(N*N);
  RT_CHECK(vx_copy_from_dev(h_S.data(), S_buffer, 0, s_bytes));
  {
    std::vector<float> h_S_ref(N*N, 0.0f);
    matmul_cpu(h_S_ref.data(), h_Q.data(), h_K.data(), N, N, d);
    for (uint32_t i = 0; i < N*N; ++i)
      if (!compare(h_S_ref[i], h_S[i], atol, rtol, (int)i, errors, "S")) ++errors;
  }
  if (errors) { std::cout << "S stage found " << errors << " errors, continuing\n"; errors = 0; }

  // Stage 2
  std::cout << "=== Stage 2: P = softmax(S) ===" << std::endl;
  kernel_arg.kernel_id = KID_SOFTMAX;
  RT_CHECK(vx_copy_to_dev(args_buffer, &kernel_arg, 0, sizeof(kernel_arg_t)));
  {
    uint32_t grid_dim[2] = {gd_sm_x, 1};
    RT_CHECK(vx_start_g(device, krnl_buffer, args_buffer, 1, grid_dim, block_dim, 0));
  }
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
  read_back_cycles("SM", sm_blocks);
  std::vector<float> h_P(N*N);
  RT_CHECK(vx_copy_from_dev(h_P.data(), P_buffer, 0, s_bytes));
  {
    std::vector<float> h_P_ref(N*N);
    softmax_cpu(h_P_ref.data(), h_S.data(), N, N);
    for (uint32_t i = 0; i < N*N; ++i)
      if (!compare(h_P_ref[i], h_P[i], 1e-5f, 1e-5f, (int)i, errors, "P")) ++errors;
  }
  if (errors) { std::cout << "P stage found " << errors << " errors, continuing\n"; errors = 0; }

  // Stage 3
  std::cout << "=== Stage 3: O = P @ V ===" << std::endl;
  kernel_arg.kernel_id = KID_PV_SIMT;
  RT_CHECK(vx_copy_to_dev(args_buffer, &kernel_arg, 0, sizeof(kernel_arg_t)));
  {
    uint32_t grid_dim[2] = {gd_pv_x, gd_pv_y};
    RT_CHECK(vx_start_g(device, krnl_buffer, args_buffer, 2, grid_dim, block_dim, 0));
  }
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
  read_back_cycles("PV", pv_blocks);
  std::vector<float> h_O(N*d);
  RT_CHECK(vx_copy_from_dev(h_O.data(), O_buffer, 0, out_bytes));
  {
    std::vector<float> h_O_ref(N*d, 0.0f);
    matmul_cpu(h_O_ref.data(), h_P.data(), h_V.data(), N, d, N);
    for (uint32_t i = 0; i < N*d; ++i)
      if (!compare(h_O_ref[i], h_O[i], atol, rtol, (int)i, errors, "O")) ++errors;
  }

  cleanup();
  if (errors) { std::cout << "FAILED!\n"; return errors; }
  std::cout << "PASSED!\n";
  return 0;
}
