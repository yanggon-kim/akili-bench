// akili_flash_tcu_fa host: fused WGMMA FlashAttention-2 forward,
// O = softmax(Q Kᵀ) V with Q, K, V [N x FA_D] (no 1/sqrt(d) scale, as in
// akili_attn), checked against an fp32 CPU reference on the fp16-rounded
// inputs.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <vortex.h>
#include "wgmma_dxa_host.h"
#include "common.h"

#define RT_CHECK(_expr)                                                     \
  do {                                                                      \
    int _ret = _expr;                                                       \
    if (0 == _ret) break;                                                   \
    printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);                \
    cleanup();                                                              \
    exit(-1);                                                               \
  } while (false)

constexpr uint32_t kDescQ = 0;
constexpr uint32_t kDescK = 1;
constexpr uint32_t kDescV = 2;
constexpr uint32_t kWarpsPerCta = FA_BR / VX_CFG_NUM_THREADS;

static vx_device_h device = nullptr;
static vx_buffer_h Q_buf = nullptr, K_buf = nullptr, V_buf = nullptr, O_buf = nullptr;

static void cleanup() {
  if (device) {
    if (Q_buf) vx_mem_free(Q_buf);
    if (K_buf) vx_mem_free(K_buf);
    if (V_buf) vx_mem_free(V_buf);
    if (O_buf) vx_mem_free(O_buf);
    vx_dev_close(device);
  }
}

static float frand() { return 10.0f * (float(rand()) / RAND_MAX) - 5.0f; }

static uint16_t f2h_host(float x) {
  uint32_t bits;
  std::memcpy(&bits, &x, sizeof(bits));
  uint16_t sign = (uint16_t)((bits >> 16) & 0x8000);
  uint16_t mant = (uint16_t)((bits >> 13) & 0x03FF);
  int32_t ex = (int32_t)((bits >> 23) & 0xFF) - 127 + 15;
  if (ex >= 31) return (uint16_t)(sign | 0x7C00);
  if (ex <= 0) return sign;
  return (uint16_t)(sign | ((uint16_t)ex << 10) | mant);
}

static float h2f_host(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000) << 16;
  uint32_t ex = (h >> 10) & 0x1F, mant = h & 0x3FF;
  uint32_t bits = ex == 0 ? sign : (sign | ((ex - 15 + 127) << 23) | (mant << 13));
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

static void attention_cpu(std::vector<float>& O, const std::vector<float>& Q,
                          const std::vector<float>& K, const std::vector<float>& V,
                          uint32_t N, uint32_t d) {
  std::vector<double> s(N);
  for (uint32_t i = 0; i < N; ++i) {
    double mx = -1e300;
    for (uint32_t j = 0; j < N; ++j) {
      double acc = 0;
      for (uint32_t k = 0; k < d; ++k) acc += (double)Q[i * d + k] * K[j * d + k];
      s[j] = acc;
      mx = acc > mx ? acc : mx;
    }
    double sum = 0;
    for (uint32_t j = 0; j < N; ++j) { s[j] = std::exp(s[j] - mx); sum += s[j]; }
    for (uint32_t k = 0; k < d; ++k) {
      double acc = 0;
      for (uint32_t j = 0; j < N; ++j) acc += s[j] * V[j * d + k];
      O[i * d + k] = (float)(acc / sum);
    }
  }
}

int main(int argc, char** argv) {
  uint32_t N = 512;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "-n") && i + 1 < argc) N = (uint32_t)atoi(argv[++i]);
  }
  if (N % FA_BR || N % FA_BC) {
    printf("Error: N=%u must be a multiple of %u\n", N, FA_BR);
    return -1;
  }
  const uint32_t d = FA_D;
  printf("akili_flash_tcu_fa: N=%u d=%u Br=%u Bc=%u\n", N, d, FA_BR, FA_BC);

  RT_CHECK(vx_dev_open(&device));

  srand(42);
  std::vector<float> h_Q(N * d), h_K(N * d), h_V(N * d);
  std::vector<uint16_t> q16(N * d), k16(N * d), v16(N * d);
  for (uint32_t i = 0; i < N * d; ++i) {
    q16[i] = f2h_host(frand()); h_Q[i] = h2f_host(q16[i]);
    k16[i] = f2h_host(frand()); h_K[i] = h2f_host(k16[i]);
    v16[i] = f2h_host(frand()); h_V[i] = h2f_host(v16[i]);
  }

  kernel_arg_t arg = {};
  uint32_t in_bytes = N * d * sizeof(uint16_t);
  RT_CHECK(vx_mem_alloc(device, in_bytes, VX_MEM_READ, &Q_buf));
  RT_CHECK(vx_mem_alloc(device, in_bytes, VX_MEM_READ, &K_buf));
  RT_CHECK(vx_mem_alloc(device, in_bytes, VX_MEM_READ, &V_buf));
  RT_CHECK(vx_mem_alloc(device, N * d * sizeof(float), VX_MEM_WRITE, &O_buf));
  RT_CHECK(vx_mem_address(Q_buf, &arg.Q_addr));
  RT_CHECK(vx_mem_address(K_buf, &arg.K_addr));
  RT_CHECK(vx_mem_address(V_buf, &arg.V_addr));
  RT_CHECK(vx_mem_address(O_buf, &arg.O_addr));
  RT_CHECK(vx_copy_to_dev(Q_buf, q16.data(), 0, in_bytes));
  RT_CHECK(vx_copy_to_dev(K_buf, k16.data(), 0, in_bytes));
  RT_CHECK(vx_copy_to_dev(V_buf, v16.data(), 0, in_bytes));
  arg.N = N;

  // Q: A operand, row-major [Br x d] per CTA.
  RT_CHECK(vx_dxa_program_desc_2d(device, kDescQ, arg.Q_addr, d, N, d * 2, d, FA_BR, 2));
  // K: B of Q Kᵀ — row-major K rows are already the K-major [Bc][d] tile.
  RT_CHECK(vx_dxa_program_desc_2d(device, kDescK, arg.K_addr, d, N, d * 2, d, FA_BC, 2));
  // V: B of P V — DXA transposes [Bc keys][d] into the K-major [d][Bc] tile.
  RT_CHECK(vx_dxa_program_desc_2d(device, kDescV, arg.V_addr, d, N, d * 2, d, FA_BC, 2));
  RT_CHECK(akili_dxa_set_kmajor(device, kDescV, 2));

  uint32_t grid[2] = {N / FA_BR, 1};
  uint32_t block[2] = {kWarpsPerCta * VX_CFG_NUM_THREADS, 1};
  auto t0 = std::chrono::high_resolution_clock::now();
  RT_CHECK(akili_start_g(device, &arg, sizeof(arg), 1, grid, block, 16384));
  auto t1 = std::chrono::high_resolution_clock::now();
  printf("FLASH: %ld us (grid=%u)\n",
         (long)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count(), grid[0]);

  std::vector<float> h_O(N * d);
  RT_CHECK(vx_copy_from_dev(h_O.data(), O_buf, 0, N * d * sizeof(float)));

  std::vector<float> ref(N * d);
  attention_cpu(ref, h_Q, h_K, h_V, N, d);
  const float atol = 2e-2f, rtol = 2e-2f;
  int errors = 0;
  for (uint32_t i = 0; i < N * d; ++i) {
    if (std::fabs(h_O[i] - ref[i]) > atol + rtol * std::fabs(ref[i])) {
      if (errors < 10)
        printf("*** O error: [%u] expected=%f, actual=%f\n", i, ref[i], h_O[i]);
      ++errors;
    }
  }

  cleanup();
  if (errors) {
    printf("Found %d / %u errors!\nFAILED!\n", errors, N * d);
    return errors;
  }
  printf("PASSED!\n");
  return 0;
}
