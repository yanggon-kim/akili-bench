// akili_attn_tcu — Dense TCU attention.
//
// Three kernel launches per run:
//   Stage 1: Q·Kᵀ    — dense TCU kernel (sgemm_tcu pattern)
//   Stage 2: softmax — SIMT fp32 row kernel
//   Stage 3: P·V     — dense TCU kernel
//
// Host prepacks:
//   Q_fp16:       [N × d]  row-major
//   K_fp16_cm:    [N × d]  row-major, K_cm[n*d+k] = K_host[k*N+n]
//   V_fp16_cm:    [d × N]  row-major, V_cm[d_idx*N+n] = V_host[n*d+d_idx]
//   P_fp16:       [N × N]  row-major (converted after softmax readback)

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
#ifndef NUM_TCU_LANES
#define NUM_TCU_LANES NUM_THREADS
#endif

#include <tensor_cfg.h>
namespace vt = vortex::tensor;
using cfg = vt::wmma_config_t<NUM_TCU_LANES, vt::fp16, vt::fp32>;
static constexpr uint32_t TM = cfg::tileM;
static constexpr uint32_t TN = cfg::tileN;
static constexpr uint32_t TK = cfg::tileK;

#define RT_CHECK(_expr)                                         \
   do {                                                         \
     int _ret = _expr;                                          \
     if (0 == _ret) break;                                      \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
     cleanup();                                                 \
     exit(-1);                                                  \
   } while (false)

static float frand() { return 10.0f * (float(rand()) / RAND_MAX) - 5.0f; }

// Manual fp32↔fp16 IEEE encoding so we don't depend on __fp16 host support.
static inline uint16_t f2h_host(float x) {
  uint32_t bits;
  std::memcpy(&bits, &x, sizeof(bits));
  uint16_t sign = (bits >> 16) & 0x8000;
  uint16_t mant = (bits >> 13) & 0x03FF;
  int32_t  exp  = (int32_t)((bits >> 23) & 0xFF) - 127 + 15;
  if (exp >= 31) return sign | 0x7C00;
  if (exp <= 0)  return sign;
  return sign | ((uint16_t)exp << 10) | mant;
}

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

// ---------------------------------------------------------------------------

const char* kernel_file = "kernel.vxbin";
uint32_t N_req = 16;
uint32_t d_req = 16;

vx_device_h device = nullptr;
vx_buffer_h Q_fp16_buffer = nullptr;
vx_buffer_h K_fp16_buffer = nullptr;
vx_buffer_h V_fp16_buffer = nullptr;
vx_buffer_h P_fp16_buffer = nullptr;
vx_buffer_h S_buffer      = nullptr;
vx_buffer_h P_fp32_buffer = nullptr;
vx_buffer_h O_buffer      = nullptr;
vx_buffer_h cycles_buffer = nullptr;
vx_buffer_h instrs_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
  std::cout << "akili_attn_tcu — Dense TCU attention benchmark" << std::endl;
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
    if (Q_fp16_buffer) vx_mem_free(Q_fp16_buffer);
    if (K_fp16_buffer) vx_mem_free(K_fp16_buffer);
    if (V_fp16_buffer) vx_mem_free(V_fp16_buffer);
    if (P_fp16_buffer) vx_mem_free(P_fp16_buffer);
    if (S_buffer)      vx_mem_free(S_buffer);
    if (P_fp32_buffer) vx_mem_free(P_fp32_buffer);
    if (O_buffer)      vx_mem_free(O_buffer);
    if (cycles_buffer) vx_mem_free(cycles_buffer);
    if (instrs_buffer) vx_mem_free(instrs_buffer);
    if (krnl_buffer)   vx_mem_free(krnl_buffer);
    if (args_buffer)   vx_mem_free(args_buffer);
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
static void read_back_instrs(const char* stage_tag, uint32_t num_blocks) {
  std::vector<uint32_t> h_instrs(num_blocks, 0);
  vx_copy_from_dev(h_instrs.data(), instrs_buffer, 0, num_blocks * sizeof(uint32_t));
  uint32_t max_ins = 0;
  for (auto i : h_instrs) if (i > max_ins) max_ins = i;
  printf("KINS[%s,nt=%u]: %u\n", stage_tag, (unsigned)NUM_THREADS, max_ins);
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

  // Pad N, d to TCU tile multiples.
  auto round_up = [](uint32_t v, uint32_t m) { return ((v + m - 1) / m) * m; };
  uint32_t n_align = std::max({TM, TN, TK});
  uint32_t d_align = std::max(TK, TN);
  uint32_t N = round_up(N_req, n_align);
  uint32_t d = round_up(d_req, d_align);
  std::cout << "akili_attn_tcu — padded N=" << N << " (req " << N_req
            << "), d=" << d << " (req " << d_req << "), "
            << "TM=" << TM << " TN=" << TN << " TK=" << TK << std::endl;

  std::vector<float> h_Q(N * d, 0.0f);
  std::vector<float> h_K(d * N, 0.0f);
  std::vector<float> h_V(N * d, 0.0f);
  for (uint32_t r = 0; r < N_req; ++r)
    for (uint32_t c = 0; c < d_req; ++c)
      h_Q[r * d + c] = frand();
  for (uint32_t r = 0; r < d_req; ++r)
    for (uint32_t c = 0; c < N_req; ++c)
      h_K[r * N + c] = frand();
  for (uint32_t r = 0; r < N_req; ++r)
    for (uint32_t c = 0; c < d_req; ++c)
      h_V[r * d + c] = frand();

  uint32_t in_fp16_bytes = N * d * sizeof(uint16_t);
  uint32_t s_fp32_bytes  = N * N * sizeof(float);
  uint32_t s_fp16_bytes  = N * N * sizeof(uint16_t);
  uint32_t out_bytes     = N * d * sizeof(float);

  RT_CHECK(vx_mem_alloc(device, in_fp16_bytes, VX_MEM_READ_WRITE, &Q_fp16_buffer));
  RT_CHECK(vx_mem_address(Q_fp16_buffer, &kernel_arg.Q_addr));
  RT_CHECK(vx_mem_alloc(device, in_fp16_bytes, VX_MEM_READ_WRITE, &K_fp16_buffer));
  RT_CHECK(vx_mem_address(K_fp16_buffer, &kernel_arg.K_addr));
  RT_CHECK(vx_mem_alloc(device, in_fp16_bytes, VX_MEM_READ_WRITE, &V_fp16_buffer));
  RT_CHECK(vx_mem_address(V_fp16_buffer, &kernel_arg.V_addr));
  RT_CHECK(vx_mem_alloc(device, s_fp32_bytes,  VX_MEM_READ_WRITE, &S_buffer));
  RT_CHECK(vx_mem_address(S_buffer, &kernel_arg.S_addr));
  RT_CHECK(vx_mem_alloc(device, s_fp32_bytes,  VX_MEM_READ_WRITE, &P_fp32_buffer));
  uint64_t P_fp32_addr = 0;
  RT_CHECK(vx_mem_address(P_fp32_buffer, &P_fp32_addr));
  RT_CHECK(vx_mem_alloc(device, s_fp16_bytes,  VX_MEM_READ_WRITE, &P_fp16_buffer));
  uint64_t P_fp16_addr = 0;
  RT_CHECK(vx_mem_address(P_fp16_buffer, &P_fp16_addr));
  RT_CHECK(vx_mem_alloc(device, out_bytes,     VX_MEM_READ_WRITE, &O_buffer));
  RT_CHECK(vx_mem_address(O_buffer, &kernel_arg.O_addr));

  kernel_arg.N = N;
  kernel_arg.d = d;

  // ---- Pack fp16 tensors host-side ----
  {
    std::vector<uint16_t> h_Q_cm(N * d);
    std::vector<uint16_t> h_K_cm(N * d);
    std::vector<uint16_t> h_V_cm(N * d);
    for (uint32_t i = 0; i < N * d; ++i) h_Q_cm[i] = f2h_host(h_Q[i]);
    for (uint32_t n = 0; n < N; ++n)
      for (uint32_t k = 0; k < d; ++k)
        h_K_cm[n * d + k] = f2h_host(h_K[k * N + n]);
    for (uint32_t d_idx = 0; d_idx < d; ++d_idx)
      for (uint32_t n = 0; n < N; ++n)
        h_V_cm[d_idx * N + n] = f2h_host(h_V[n * d + d_idx]);
    RT_CHECK(vx_copy_to_dev(Q_fp16_buffer, h_Q_cm.data(), 0, in_fp16_bytes));
    RT_CHECK(vx_copy_to_dev(K_fp16_buffer, h_K_cm.data(), 0, in_fp16_bytes));
    RT_CHECK(vx_copy_to_dev(V_fp16_buffer, h_V_cm.data(), 0, in_fp16_bytes));
  }

  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
  RT_CHECK(vx_mem_alloc(device, sizeof(kernel_arg_t), VX_MEM_READ_WRITE, &args_buffer));

  // Compute grid sizes per stage and allocate max-sized cycles buffer.
  uint32_t qk_blocks = (N / TN) * (N / TM);
  uint32_t sm_blocks = (N + NUM_THREADS - 1) / NUM_THREADS;
  uint32_t pv_blocks = (d / TN) * (N / TM);
  uint32_t max_blocks = qk_blocks;
  if (sm_blocks > max_blocks) max_blocks = sm_blocks;
  if (pv_blocks > max_blocks) max_blocks = pv_blocks;
  RT_CHECK(vx_mem_alloc(device, max_blocks * sizeof(uint32_t),
                        VX_MEM_READ_WRITE, &cycles_buffer));
  RT_CHECK(vx_mem_address(cycles_buffer, &kernel_arg.cycles_addr));
  RT_CHECK(vx_mem_alloc(device, max_blocks * sizeof(uint32_t),
                        VX_MEM_READ_WRITE, &instrs_buffer));
  RT_CHECK(vx_mem_address(instrs_buffer, &kernel_arg.instrs_addr));

  int errors = 0;
  const float atol = 2e-2f;
  const float rtol = 2e-2f;

  uint32_t block_dim[2] = {NUM_THREADS, 1};

  // ---------------- Stage 1: S = Q @ K^T (dense TCU) ----------------
  std::cout << "=== Stage 1: S = Q @ K^T (dense TCU) ===" << std::endl;
  kernel_arg.kernel_id = KID_QK_TCU;
  RT_CHECK(vx_copy_to_dev(args_buffer, &kernel_arg, 0, sizeof(kernel_arg_t)));
  {
    uint32_t grid_dim[2] = {N / TN, N / TM};
    RT_CHECK(vx_start_g(device, krnl_buffer, args_buffer, 2, grid_dim, block_dim, 0));
  }
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
  read_back_cycles("QK", qk_blocks); read_back_instrs("QK", qk_blocks);

  std::vector<float> h_S(N * N, 0.0f);
  RT_CHECK(vx_copy_from_dev(h_S.data(), S_buffer, 0, s_fp32_bytes));
  {
    std::vector<float> h_S_ref(N * N, 0.0f);
    matmul_cpu(h_S_ref.data(), h_Q.data(), h_K.data(), N, N, d);
    for (uint32_t r = 0; r < N_req; ++r)
      for (uint32_t c = 0; c < N_req; ++c) {
        int idx = r * N + c;
        if (!compare(h_S_ref[idx], h_S[idx], atol, rtol, idx, errors, "S"))
          ++errors;
      }
  }
  if (errors) {
    std::cout << "S stage found " << errors << " errors, continuing" << std::endl;
    errors = 0;
  }

  // Zero out the padded rows/cols in S so softmax behaves.
  for (uint32_t r = 0; r < N; ++r)
    for (uint32_t c = 0; c < N; ++c)
      if (r >= N_req || c >= N_req) h_S[r * N + c] = -1e9f;
  RT_CHECK(vx_copy_to_dev(S_buffer, h_S.data(), 0, s_fp32_bytes));

  // ---------------- Stage 2: P = softmax(S) — SIMT fp32 ----------------
  std::cout << "=== Stage 2: P = softmax(S) ===" << std::endl;
  kernel_arg.P_addr    = P_fp32_addr;
  kernel_arg.kernel_id = KID_SOFTMAX;
  RT_CHECK(vx_copy_to_dev(args_buffer, &kernel_arg, 0, sizeof(kernel_arg_t)));
  {
    uint32_t grid_dim[2] = {sm_blocks, 1};
    RT_CHECK(vx_start_g(device, krnl_buffer, args_buffer, 1, grid_dim, block_dim, 0));
  }
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
  read_back_cycles("SM", sm_blocks); read_back_instrs("SM", sm_blocks);

  std::vector<float> h_P(N * N);
  RT_CHECK(vx_copy_from_dev(h_P.data(), P_fp32_buffer, 0, s_fp32_bytes));
  {
    std::vector<float> h_P_ref(N * N);
    softmax_cpu(h_P_ref.data(), h_S.data(), N, N);
    for (uint32_t r = 0; r < N_req; ++r)
      for (uint32_t c = 0; c < N_req; ++c) {
        int idx = r * N + c;
        if (!compare(h_P_ref[idx], h_P[idx], 1e-5f, 1e-5f, idx, errors, "P"))
          ++errors;
      }
  }
  if (errors) {
    std::cout << "P stage found " << errors << " errors, continuing" << std::endl;
    errors = 0;
  }

  // ---------------- Stage 3: O = P @ V (dense TCU) ----------------
  std::cout << "=== Stage 3: O = P @ V (dense TCU) ===" << std::endl;
  {
    std::vector<uint16_t> h_P_fp16(N * N);
    for (uint32_t i = 0; i < N * N; ++i) h_P_fp16[i] = f2h_host(h_P[i]);
    RT_CHECK(vx_copy_to_dev(P_fp16_buffer, h_P_fp16.data(), 0, s_fp16_bytes));
  }
  kernel_arg.P_addr    = P_fp16_addr;
  kernel_arg.kernel_id = KID_PV_TCU;
  RT_CHECK(vx_copy_to_dev(args_buffer, &kernel_arg, 0, sizeof(kernel_arg_t)));
  {
    uint32_t grid_dim[2] = {d / TN, N / TM};
    RT_CHECK(vx_start_g(device, krnl_buffer, args_buffer, 2, grid_dim, block_dim, 0));
  }
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
  read_back_cycles("PV", pv_blocks); read_back_instrs("PV", pv_blocks);

  std::vector<float> h_O(N * d);
  RT_CHECK(vx_copy_from_dev(h_O.data(), O_buffer, 0, out_bytes));
  {
    std::vector<float> h_O_ref(N * d, 0.0f);
    matmul_cpu(h_O_ref.data(), h_P.data(), h_V.data(), N, d, N);
    for (uint32_t r = 0; r < N_req; ++r)
      for (uint32_t c = 0; c < d_req; ++c) {
        int idx = r * d + c;
        if (!compare(h_O_ref[idx], h_O[idx], atol, rtol, idx, errors, "O"))
          ++errors;
      }
  }

  cleanup();

  if (errors != 0) {
    std::cout << "Found " << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return errors;
  }
  std::cout << "PASSED!" << std::endl;
  return 0;
}
