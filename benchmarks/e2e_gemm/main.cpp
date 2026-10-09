#include <unistd.h>
#include "e2e_host.h"
#include "common.h"
using namespace e2e;

int main(int argc, char** argv) {
  uint32_t M = 384, N = 128, K = 1152;
  int c;
  while ((c = getopt(argc, argv, "m:n:k:")) != -1) {
    if (c == 'm') M = atoi(optarg);
    if (c == 'n') N = atoi(optarg);
    if (c == 'k') K = atoi(optarg);
  }
#if E2E_TCU
  const uint32_t BM = GEMM_BM, BN = GEMM_BN, num_kt = K / KT;
#else
  const uint32_t BM = 128, BN = 128, num_kt = K / 16;
#endif
  std::printf("E2E_APP app=gemm M=%u N=%u K=%u\n", M, N, K);
  Runtime rt("kernel.vxbin");
  Rng rng(3);
  std::vector<float> a((size_t)M * K), b((size_t)N * K);
  for (auto& v : a) v = rng.next();
  for (auto& v : b) v = rng.next();
  QTensor aq = quantize_host(a.data(), M, K, tensor_scale_for(1.0f), BM);
  QTensor bq = quantize_host(b.data(), N, K, tensor_scale_for(1.0f), BN);
  gemm_args_t g = {};
  g.a = rt.upload(aq.words); g.b = rt.upload(bq.words);
  if (IS_MX) { g.a_sc = rt.upload(aq.scales); g.b_sc = rt.upload(bq.scales); }
  g.c = rt.alloc((uint64_t)M * N * 4);
  g.M = M; g.N = N; g.K = K; g.num_kt = num_kt;
#if E2E_TCU
  const uint32_t block = 1024, lmem = tcu_gemm_lmem(BM, BN);
  const char* names[] = {"g_full"};
#else
  rt.desc_blocked(0, g.b, N, num_kt, SLDW_SIMT, BN);
  rt.desc_blocked(1, g.a, M, num_kt, SLDW_SIMT, BM);
  const uint32_t block = 1024, lmem = simt_gemm_smem(BM, BN, 2);
  const char* names[] = {"g_full"};
#endif
  int idx = 0;
  for (auto n : names) {
    std::string nm = std::to_string(idx++) + "_" + n;
    rt.launch2(nm.c_str(), rt.kernel(n), g, N / BN, M / BM, block, lmem);
  }

  std::vector<float> h_c((size_t)M * N);
  rt.read(h_c.data(), g.c, h_c.size() * 4);
  std::vector<float> ref((size_t)M * N);
  for (uint32_t i = 0; i < M; ++i)
    for (uint32_t j = 0; j < N; ++j) {
      double acc = 0;
      for (uint32_t k = 0; k < K; ++k) acc += (double)aq.deq[(size_t)i * K + k] * bq.deq[(size_t)j * K + k];
      ref[(size_t)i * N + j] = (float)(acc / ((double)aq.tscale * bq.tscale));
    }
  double err = rel_rmse(h_c.data(), ref.data(), ref.size());
  double flops = 2.0 * M * N * K, bytes = (double)(aq.words.size() * (N / BN) + bq.words.size() * (M / BM)) * 4;
  for (auto& n : rt.order)
    std::printf("GEMM %s cycles=%lu flop_per_cycle=%.1f bytes_per_cycle=%.2f\n", n.c_str(),
                (unsigned long)rt.phases[n].v[0], flops / rt.phases[n].v[0], bytes / rt.phases[n].v[0]);
  rt.report(err, err, err < 1e-3);
  return 0;
}
