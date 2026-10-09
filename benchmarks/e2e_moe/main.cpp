#include <unistd.h>
#include <algorithm>
#include "e2e_host.h"
#include "e2e_linear.h"
#include "common.h"

using namespace e2e;

static uint32_t T = 512, D = 256, FF = 512, E = 8;
static const char* kernel_file = "kernel.vxbin";

#if E2E_TCU
static const uint32_t WRB = 64;
#else
static const uint32_t WRB = 128;
#endif

static float amax(const std::vector<float>& v) {
  float m = 0;
  for (float x : v) m = std::max(m, std::fabs(x));
  return m;
}

static float silu(float g) { return g / (1.0f + std::exp(-g)); }

int main(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "t:d:f:e:k:")) != -1) {
    if (c == 't') T = atoi(optarg);
    if (c == 'd') D = atoi(optarg);
    if (c == 'f') FF = atoi(optarg);
    if (c == 'e') E = atoi(optarg);
    if (c == 'k') kernel_file = optarg;
  }
  if (E > MOE_MAX_E || D % 128 || FF % 128) { std::printf("bad shape\n"); return -1; }
  const uint32_t K = MOE_TOPK;
  const uint32_t mpad = round_up(T * K + E * (LIN_BM - 1), LIN_BM);

  // ---- data: x, router, experts (W1 gate, W3 up: [FF][D]; W2 down: [D][FF]) ----
  Rng rng(5);
  std::vector<float> x((size_t)T * D), wg((size_t)E * D);
  for (auto& v : x) v = rng.next();
  for (auto& v : wg) v = rng.next() * std::sqrt(6.0f / D);
  std::vector<std::vector<float>> w1(E), w3(E), w2(E);
  for (uint32_t e = 0; e < E; ++e) {
    w1[e].resize((size_t)FF * D); w3[e].resize((size_t)FF * D); w2[e].resize((size_t)D * FF);
    for (auto& v : w1[e]) v = rng.next() * std::sqrt(6.0f / D);
    for (auto& v : w3[e]) v = rng.next() * std::sqrt(6.0f / D);
    for (auto& v : w2[e]) v = rng.next() * std::sqrt(6.0f / FF);
  }
  // Grouped weights: W13 rows interleave gate/up in SWIGLU_G groups per expert.
  std::vector<float> w13((size_t)E * 2 * FF * D), w2all((size_t)E * D * FF);
  for (uint32_t e = 0; e < E; ++e) {
    for (uint32_t j = 0; j < FF; ++j) {
      uint32_t g = (j / SWIGLU_G) * 2 * SWIGLU_G + j % SWIGLU_G;
      std::copy(&w1[e][(size_t)j * D], &w1[e][(size_t)j * D] + D, &w13[((size_t)e * 2 * FF + g) * D]);
      std::copy(&w3[e][(size_t)j * D], &w3[e][(size_t)j * D] + D, &w13[((size_t)e * 2 * FF + g + SWIGLU_G) * D]);
    }
    std::copy(w2[e].begin(), w2[e].end(), &w2all[(size_t)e * D * FF]);
  }

  // ---- routing (fp32, shared by references) ----
  std::vector<uint32_t> tid(T * K);
  std::vector<float> tw(T * K);
  for (uint32_t t = 0; t < T; ++t) {
    std::vector<float> l(E);
    float m = -1e30f, s = 0;
    for (uint32_t e = 0; e < E; ++e) {
      double a = 0;
      for (uint32_t k = 0; k < D; ++k) a += (double)x[(size_t)t * D + k] * wg[(size_t)e * D + k];
      l[e] = (float)a;
      m = std::max(m, l[e]);
    }
    for (auto& v : l) { v = std::exp(v - m); s += v; }
    float ws = 0;
    for (uint32_t k = 0; k < K; ++k) {
      uint32_t b = (uint32_t)(std::max_element(l.begin(), l.end()) - l.begin());
      tid[t * K + k] = b;
      tw[t * K + k] = l[b] / s;
      ws += tw[t * K + k];
      l[b] = -1.0f;
    }
    for (uint32_t k = 0; k < K; ++k) tw[t * K + k] /= ws;
  }

  // ---- references: fp32 and device-emulated ----
  const float ts_w13 = tensor_scale_for(amax(w13)), ts_w2 = tensor_scale_for(amax(w2all));
  QTensor w13q = quantize_host(w13.data(), E * 2 * FF, D, ts_w13, WRB);
  QTensor w2q = quantize_host(w2all.data(), E * D, FF, ts_w2, WRB);
  float h_amax = 0;
  auto moe = [&](bool emulate, float ts_x, float ts_h) {
    const std::vector<float>& W13 = emulate ? w13q.deq : w13;
    const std::vector<float>& W2 = emulate ? w2q.deq : w2all;
    std::vector<float> xs = emulate ? quantize_host(x.data(), T, D, ts_x, 16).deq : x;
    std::vector<float> y((size_t)T * D, 0.0f), h(FF);
    for (uint32_t t = 0; t < T; ++t)
      for (uint32_t k = 0; k < K; ++k) {
        uint32_t e = tid[t * K + k];
        for (uint32_t j = 0; j < FF; ++j) {
          uint32_t g = (j / SWIGLU_G) * 2 * SWIGLU_G + j % SWIGLU_G;
          double a = 0, b = 0;
          const float* wgr = &W13[((size_t)e * 2 * FF + g) * D];
          const float* wur = &W13[((size_t)e * 2 * FF + g + SWIGLU_G) * D];
          for (uint32_t i = 0; i < D; ++i) { a += (double)xs[(size_t)t * D + i] * wgr[i]; b += (double)xs[(size_t)t * D + i] * wur[i]; }
          h[j] = silu((float)a) * (float)b * tw[t * K + k];
          h_amax = std::max(h_amax, std::fabs(h[j]));
        }
        if (emulate) quant_dequant_row(h.data(), FF, ts_h, h.data());
        for (uint32_t o = 0; o < D; ++o) {
          double a = 0;
          const float* wr = &W2[((size_t)e * D + o) * FF];
          for (uint32_t j = 0; j < FF; ++j) a += (double)h[j] * wr[j];
          y[(size_t)t * D + o] += (float)a;
        }
      }
    return y;
  };
  std::vector<float> y_ref = moe(false, 1.0f, 1.0f);
  const float ts_x = tensor_scale_for(amax(x)), ts_h = tensor_scale_for(h_amax);
#if E2E_TCU
  std::vector<float> y_emu = moe(true, ts_x, ts_h);
#else
  std::vector<float> y_emu = y_ref;
#endif
  std::vector<uint32_t> cnt(E, 0);
  for (auto e : tid) ++cnt[e];
  std::printf("E2E_APP app=moe tokens=%u d_model=%u d_ff=%u experts=%u topk=%u mpad=%u expert_gemm_flops=%.0f routed_counts=",
              T, D, FF, E, K, mpad, 2.0 * T * K * 3.0 * D * FF);
  for (auto v : cnt) std::printf("%u,", v);
  std::printf("\n");

  // ---- device ----
  Runtime rt(kernel_file);
  auto k_router = rt.kernel("router"), k_align = rt.kernel("align"), k_gather = rt.kernel("gather_quant");
  auto k_lin = rt.kernel("linear"), k_comb = rt.kernel("combine");
  moe_args_t ma = {};
  ma.x = rt.upload(x); ma.wg = rt.upload(wg);
  ma.topk_id = rt.alloc(T * K * 4); ma.topk_w = rt.alloc(T * K * 4);
  ma.sorted_tok = rt.alloc(mpad * 4); ma.row_w = rt.alloc(mpad * 4); ma.pos = rt.alloc(T * K * 4);
  ma.tile_w13 = rt.alloc(mpad / LIN_BM * 4); ma.tile_w2 = rt.alloc(mpad / LIN_BM * 4);
  ma.a = rt.alloc((uint64_t)mpad * D / F::epw * 4);
  ma.a_sc = IS_MX ? rt.alloc((uint64_t)(mpad / 16) * (D / KT) * SC_TILE_BYTES) : 0;
  uint64_t hbuf = rt.alloc((uint64_t)mpad * FF / F::epw * 4);
  uint64_t hsc = IS_MX ? rt.alloc((uint64_t)(mpad / 16) * (FF / KT) * SC_TILE_BYTES) : 0;
  ma.y_perm = rt.alloc((uint64_t)mpad * D * 4); ma.y = rt.alloc((uint64_t)T * D * 4);
  ma.ts_x = ts_x; ma.T = T; ma.d = D; ma.ffn = FF; ma.E = E; ma.mpad = mpad;
  uint64_t d_w13 = rt.upload(w13q.words), d_w2 = rt.upload(w2q.words);
  uint64_t d_w13s = IS_MX ? rt.upload(w13q.scales) : 0, d_w2s = IS_MX ? rt.upload(w2q.scales) : 0;

  const uint32_t fill_block = 256, fill_grid = VX_CFG_NUM_WARPS * VX_CFG_NUM_THREADS / fill_block;
#if E2E_TCU
  const uint32_t lmem = tcu_gemm_lmem(LIN_BM, LIN_BN);
#else
  const uint32_t lmem = simt_gemm_smem(LIN_BM, LIN_BN, 2);
#endif
  rt.launch1("router", k_router, ma, (T + fill_block - 1) / fill_block, fill_block);
  rt.launch1("align", k_align, ma, 1, 32, 64);
  rt.launch1("gather_quant", k_gather, ma, fill_grid, fill_block);

  linear_args_t g1 = {};
  g1.a = ma.a; g1.a_sc = ma.a_sc; g1.w = d_w13; g1.w_sc = d_w13s;
  g1.out = hbuf; g1.out_sc = hsc; g1.ts_a = ts_x; g1.ts_w = ts_w13; g1.ts_out = ts_h;
  g1.row_scale = ma.row_w;
  g1.M = mpad; g1.N = 2 * FF; g1.K = D; g1.num_kt = D / KT;
  g1.act = ACT_SWIGLU; g1.out_mode = OUT_QUANT; g1.ldo = FF; g1.col0 = 0;
  g1.tile_w_row0 = ma.tile_w13;
  rt.launch2("expert_gemm1_swiglu", k_lin, g1, 2 * FF / LIN_BN, mpad / LIN_BM, 1024, lmem);

  linear_args_t g2 = {};
  g2.a = hbuf; g2.a_sc = hsc; g2.w = d_w2; g2.w_sc = d_w2s;
  g2.out = ma.y_perm; g2.ts_a = ts_h; g2.ts_w = ts_w2; g2.ts_out = 1.0f;
  g2.M = mpad; g2.N = D; g2.K = FF; g2.num_kt = FF / KT;
  g2.act = ACT_NONE; g2.out_mode = OUT_FP32; g2.ldo = D; g2.col0 = 0;
  g2.tile_w_row0 = ma.tile_w2;
  rt.launch2("expert_gemm2", k_lin, g2, D / LIN_BN, mpad / LIN_BM, 1024, lmem);
  rt.launch1("combine", k_comb, ma, fill_grid, fill_block);

  std::vector<float> y((size_t)T * D);
  rt.read(y.data(), ma.y, y.size() * 4);
  double e_emu = rel_rmse(y.data(), y_emu.data(), y.size());
  double e_ref = rel_rmse(y.data(), y_ref.data(), y.size());
  rt.report(e_emu, e_ref, e_emu < 2e-2);
  return e_emu < 2e-2 ? 0 : 1;
}
