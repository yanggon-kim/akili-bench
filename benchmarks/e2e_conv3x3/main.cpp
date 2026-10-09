#include <unistd.h>
#include <algorithm>
#include "e2e_host.h"
#include "common.h"

using namespace e2e;

static uint32_t H = 32, W = 32, C = 128;
static const char* kernel_file = "kernel.vxbin";

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "h:w:c:k:")) != -1) {
    switch (c) {
    case 'h': H = atoi(optarg); break;
    case 'w': W = atoi(optarg); break;
    case 'c': C = atoi(optarg); break;
    case 'k': kernel_file = optarg; break;
    default:
      std::printf("Usage: [-h H] [-w W] [-c C] [-k kernel]\n");
      std::exit(-1);
    }
  }
}

// out[p][co] = relu(sum_{tap,ci} in[nbr(p,tap)][ci] * w[co][tap*C+ci] + b[co] (+ resid))
static void conv_ref(std::vector<float>& out, const std::vector<float>& in, const std::vector<float>& w,
                     const std::vector<float>& b, const float* resid) {
  const uint32_t P = H * W, K = 9 * C;
  out.assign((size_t)P * C, 0.0f);
  std::vector<float> col(K);
  for (uint32_t p = 0; p < P; ++p) {
    for (uint32_t tap = 0; tap < 9; ++tap) {
      int iy = (int)(p / W) + (int)(tap / 3) - 1, ix = (int)(p % W) + (int)(tap % 3) - 1;
      bool in_b = iy >= 0 && iy < (int)H && ix >= 0 && ix < (int)W;
      for (uint32_t ci = 0; ci < C; ++ci)
        col[tap * C + ci] = in_b ? in[((size_t)iy * W + ix) * C + ci] : 0.0f;
    }
    for (uint32_t co = 0; co < C; ++co) {
      double s = 0;
      const float* wr = &w[(size_t)co * K];
      for (uint32_t k = 0; k < K; ++k) s += (double)col[k] * wr[k];
      float t = (float)s + b[co];
      if (resid) t += resid[(size_t)p * C + co];
      out[(size_t)p * C + co] = t > 0.0f ? t : 0.0f;
    }
  }
}

static float amax(const std::vector<float>& v) {
  float m = 0;
  for (float x : v) m = std::max(m, std::fabs(x));
  return m;
}

int main(int argc, char** argv) {
  parse_args(argc, argv);
  const uint32_t P = H * W, K = 9 * C;
#if E2E_TCU
  const uint32_t BM = CONV_BM, BN = CONV_BN;
  if (C != CONV_C || P % BM) { std::printf("C must be a multiple of %u and 64, P of %u\n", KT, BM); return -1; }
  const uint32_t num_kt = K / KT;
#else
  if (C != CONV_C || P % 128) { std::printf("C and P must be multiples of 128\n"); return -1; }
  const uint32_t num_kt = K / 16;
#endif
  std::printf("E2E_APP app=conv3x3 H=%u W=%u C=%u P=%u K=%u gemm_flops=%.0f\n",
              H, W, C, P, K, 2.0 * 2.0 * P * C * K);

  // ---- data ----
  Rng rng(7);
  std::vector<float> x((size_t)P * C), w1((size_t)C * K), w2((size_t)C * K), b1(C), b2(C);
  for (auto& v : x) v = rng.next();
  float wa = std::sqrt(6.0f / K);
  for (auto& v : w1) v = rng.next() * wa;
  for (auto& v : w2) v = rng.next() * wa;
  for (auto& v : b1) v = rng.next() * 0.1f;
  for (auto& v : b2) v = rng.next() * 0.1f;

  // ---- fp32 reference and calibration ----
  std::vector<float> h_ref, y_ref;
  conv_ref(h_ref, x, w1, b1, nullptr);
  conv_ref(y_ref, h_ref, w2, b2, x.data());
  const float ts_x = tensor_scale_for(amax(x)), ts_h = tensor_scale_for(amax(h_ref));
  const float ts_w1 = tensor_scale_for(amax(w1)), ts_w2 = tensor_scale_for(amax(w2));

  // ---- bit-emulated reference of the quantized pipeline ----
  std::vector<float> y_emu;
#if E2E_TCU
  QTensor w1q = quantize_host(w1.data(), C, K, ts_w1, CONV_BRB);
  QTensor w2q = quantize_host(w2.data(), C, K, ts_w2, CONV_BRB);
  {
    QTensor xq = quantize_host(x.data(), P, C, ts_x, 16);
    std::vector<float> h_emu;
    conv_ref(h_emu, xq.deq, w1q.deq, b1, nullptr);
    QTensor hq = quantize_host(h_emu.data(), P, C, ts_h, 16);
    conv_ref(y_emu, hq.deq, w2q.deq, b2, x.data());
  }
#else
  y_emu = y_ref;
#endif

  // ---- device ----
  Runtime rt(kernel_file);
  auto k_im2col = rt.kernel("im2col");
  auto k_gemm = rt.kernel("gemm");
  const uint32_t EPW = F::epw;
  uint64_t d_x = rt.upload(x);
  uint64_t d_b1 = rt.upload(b1), d_b2 = rt.upload(b2);
  uint64_t d_a = rt.alloc((uint64_t)P * K / EPW * 4);
  uint64_t d_h = rt.alloc((uint64_t)P * C * 4);
  uint64_t d_y = rt.alloc((uint64_t)P * C * 4);
  uint64_t d_a_sc = 0, d_xq = 0, d_xq_sc = 0, d_h_sc = 0, d_w1 = 0, d_w2 = 0, d_w1_sc = 0, d_w2_sc = 0;

  const uint32_t fill_block = 256;
  const uint32_t fill_grid = VX_CFG_NUM_WARPS * VX_CFG_NUM_THREADS / fill_block;

  conv_args_t a = {};
  a.H = H; a.W = W; a.C = C; a.P = P; a.K = K; a.num_kt = num_kt;

#if E2E_TCU
  d_w1 = rt.upload(w1q.words);
  d_w2 = rt.upload(w2q.words);
  d_xq = rt.alloc((uint64_t)P * C / EPW * 4);
  if constexpr (IS_MX) {
    d_w1_sc = rt.upload(w1q.scales);
    d_w2_sc = rt.upload(w2q.scales);
    d_a_sc = rt.alloc((uint64_t)(P / 16) * num_kt * SC_TILE_BYTES);
    d_xq_sc = rt.alloc((uint64_t)P * C / BLK);
    d_h_sc = rt.alloc((uint64_t)P * C / BLK);
    rt.zero(d_a_sc, (uint64_t)(P / 16) * num_kt * SC_TILE_BYTES);
  }
  rt.desc_blocked(0, d_w1, C, num_kt, LDW_TCU, BN);
  rt.desc_blocked(2, d_w2, C, num_kt, LDW_TCU, BN);
  if constexpr (IS_MX) {
    rt.desc_blocked_scales(1, d_w1_sc, C, num_kt, BN);
    rt.desc_blocked_scales(3, d_w2_sc, C, num_kt, BN);
  }
  auto k_quant = rt.kernel("quant_in");
  const uint32_t gemm_block = (BM / 16) * (BN / 16) * VX_CFG_NUM_THREADS;
  const uint32_t gemm_lmem = tcu_gemm_lmem(BM, BN);

  // layer 1
  a.src = d_x; a.dst = d_xq; a.dst_sc = d_xq_sc; a.ts_out = ts_x;
  rt.launch1("quant_in", k_quant, a, fill_grid, fill_block);
  a.src = d_xq; a.src_sc = d_xq_sc; a.dst = d_a; a.dst_sc = d_a_sc;
  rt.launch1("im2col", k_im2col, a, fill_grid, fill_block);
  a.a = d_a; a.a_sc = d_a_sc; a.w = d_w1; a.w_sc = d_w1_sc;
  a.dst = d_h; a.dst_sc = d_h_sc; a.bias = d_b1; a.resid = 0;
  a.ts_a = ts_x; a.ts_w = ts_w1; a.ts_out = ts_h; a.slot = 0; a.quant_out = 1;
  rt.launch2("gemm", k_gemm, a, C / BN, P / BM, gemm_block, gemm_lmem);
  // layer 2
  a.src = d_h; a.src_sc = d_h_sc; a.dst = d_a; a.dst_sc = d_a_sc;
  rt.launch1("im2col", k_im2col, a, fill_grid, fill_block);
  a.w = d_w2; a.w_sc = d_w2_sc;
  a.dst = d_y; a.dst_sc = 0; a.bias = d_b2; a.resid = d_x;
  a.ts_a = ts_h; a.ts_w = ts_w2; a.slot = 2; a.quant_out = 0;
  rt.launch2("gemm", k_gemm, a, C / BN, P / BM, gemm_block, gemm_lmem);
#else
  (void)ts_x; (void)ts_h; (void)ts_w1; (void)ts_w2;
  (void)d_a_sc; (void)d_xq; (void)d_xq_sc; (void)d_h_sc; (void)d_w1_sc; (void)d_w2_sc;
  QTensor w1q = quantize_host(w1.data(), C, K, 1.0f, CONV_BRB);
  QTensor w2q = quantize_host(w2.data(), C, K, 1.0f, CONV_BRB);
  d_w1 = rt.upload(w1q.words);
  d_w2 = rt.upload(w2q.words);
  rt.desc_blocked(0, d_w1, C, num_kt, SLDW_SIMT, 128);
  rt.desc_blocked(1, d_w2, C, num_kt, SLDW_SIMT, 128);
  const uint32_t gemm_block = 1024, gemm_lmem = simt_gemm_smem(128, 128, 2);
  a.a = d_a;
  a.src = d_x; a.dst = d_a;
  rt.launch1("im2col", k_im2col, a, fill_grid, fill_block);
  a.w = d_w1; a.dst = d_h; a.bias = d_b1; a.resid = 0; a.slot = 0;
  rt.launch2("gemm", k_gemm, a, C / 128, P / 128, gemm_block, gemm_lmem);
  a.src = d_h; a.dst = d_a;
  rt.launch1("im2col", k_im2col, a, fill_grid, fill_block);
  a.w = d_w2; a.dst = d_y; a.bias = d_b2; a.resid = d_x; a.slot = 1;
  rt.launch2("gemm", k_gemm, a, C / 128, P / 128, gemm_block, gemm_lmem);
#endif

  std::vector<float> y((size_t)P * C);
  rt.read(y.data(), d_y, y.size() * 4);
  double e_emu = rel_rmse(y.data(), y_emu.data(), y.size());
  double e_ref = rel_rmse(y.data(), y_ref.data(), y.size());
  bool pass = e_emu < 2e-2;
  rt.report(e_emu, e_ref, pass);
  return pass ? 0 : 1;
}
