#include <unistd.h>
#include <algorithm>
#include "e2e_host.h"
#include "common.h"

using namespace e2e;

static uint32_t H = 2, N = 768;
static const char* kernel_file = "kernel.vxbin";

static float amax(const std::vector<float>& v) {
  float m = 0;
  for (float x : v) m = std::max(m, std::fabs(x));
  return m;
}

int main(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "H:n:k:")) != -1) {
    if (c == 'H') H = atoi(optarg);
    if (c == 'n') N = atoi(optarg);
    if (c == 'k') kernel_file = optarg;
  }
  const uint32_t D = FA_D;
#if E2E_TCU
  const uint32_t BM = FA_BM, BN = KT;
#else
  const uint32_t BM = FA_SIMT_BM, BN = FA_SIMT_BN;
#endif
  if (N % BM || N % BN || N % 128) { std::printf("N must be a multiple of %u, %u and 128\n", BM, BN); return -1; }
  const float qk_scale = 1.44269504f / std::sqrt((float)D);
  std::printf("E2E_APP app=flash_attn heads=%u seq=%u head_dim=%u block_m=%u block_n=%u attn_flops=%.0f\n",
              H, N, D, BM, BN, 4.0 * H * N * N * D);

  // ---- inputs Q, K, V: [H][N][D] ----
  Rng rng(9);
  std::vector<float> q((size_t)H * N * D), k((size_t)H * N * D), v((size_t)H * N * D);
  for (auto& x : q) x = 2.0f * rng.next();
  for (auto& x : k) x = 2.0f * rng.next();
  for (auto& x : v) x = rng.next();
  std::vector<float> vt((size_t)H * D * N);   // V^T per head
  for (uint32_t h = 0; h < H; ++h)
    for (uint32_t n = 0; n < N; ++n)
      for (uint32_t d = 0; d < D; ++d) vt[((size_t)h * D + d) * N + n] = v[((size_t)h * N + n) * D + d];

  const float ts_q = tensor_scale_for(amax(q)), ts_k = tensor_scale_for(amax(k));
  const float ts_v = tensor_scale_for(amax(v)), ts_p = tensor_scale_for(1.0f);
#if E2E_TCU
  QTensor qq = quantize_host(q.data(), H * N, D, ts_q, 16);
  QTensor kq = quantize_host(k.data(), H * N, D, ts_k, 16);
  QTensor vq = quantize_host(vt.data(), H * D, N, ts_v, 128);
  const std::vector<float>& qd = qq.deq;
  const std::vector<float>& kd = kq.deq;
  const std::vector<float>& vtd = vq.deq;
#else
  const std::vector<float>& qd = q;
  const std::vector<float>& kd = k;
  const std::vector<float>& vtd = vt;
#endif

  // ---- references: exact softmax (fp32 inputs) and the device's blockwise
  //      online softmax with quantized P (quantized inputs) ----
  std::vector<float> o_ref((size_t)H * N * D), o_emu((size_t)H * N * D);
  std::vector<float> s(N), p(BN);
  std::vector<uint32_t> pw(BN);
  std::vector<uint8_t> psc(BN / BLK + 1);
  for (uint32_t h = 0; h < H; ++h)
    for (uint32_t i = 0; i < N; ++i) {
      const float* qi = &q[((size_t)h * N + i) * D];
      float m = -1e30f;
      for (uint32_t j = 0; j < N; ++j) {
        double a = 0;
        const float* kj = &k[((size_t)h * N + j) * D];
        for (uint32_t d = 0; d < D; ++d) a += (double)qi[d] * kj[d];
        s[j] = (float)a * qk_scale;
        m = std::max(m, s[j]);
      }
      double l = 0;
      std::vector<double> acc(D, 0.0);
      for (uint32_t j = 0; j < N; ++j) {
        double e = std::exp2((double)s[j] - m);
        l += e;
        for (uint32_t d = 0; d < D; ++d) acc[d] += e * v[((size_t)h * N + j) * D + d];
      }
      for (uint32_t d = 0; d < D; ++d) o_ref[((size_t)h * N + i) * D + d] = (float)(acc[d] / l);
      // emulation
      const float* qe = &qd[((size_t)h * N + i) * D];
      float me = -1e30f, le = 0.0f;
      std::vector<float> oe(D, 0.0f);
      for (uint32_t j0 = 0; j0 < N; j0 += BN) {
        float mx = -1e30f;
        for (uint32_t j = 0; j < BN; ++j) {
          double a = 0;
          const float* kj = &kd[((size_t)h * N + j0 + j) * D];
          for (uint32_t d = 0; d < D; ++d) a += (double)qe[d] * kj[d];
          s[j] = (float)a * qk_scale;
          mx = std::max(mx, s[j]);
        }
        const float mn = std::max(me, mx);
        float sum = 0.0f;
        for (uint32_t j = 0; j < BN; ++j) { p[j] = exp2_shared(s[j] - mn); sum += p[j]; }
        const float alpha = exp2_shared(me - mn);
        me = mn;
        le = le * alpha + sum;
#if E2E_TCU
        quantize_run(p.data(), BN, ts_p, pw.data(), psc.data());
        const uint32_t mask = F::bits == 32 ? 0xffffffffu : ((1u << F::bits) - 1);
        for (uint32_t j = 0; j < BN; ++j)
          p[j] = decode((pw[j / F::epw] >> ((j % F::epw) * F::bits)) & mask, IS_MX ? psc[j / BLK] : 0, ts_p);
#endif
        for (uint32_t d = 0; d < D; ++d) {
          double a = 0;
          for (uint32_t j = 0; j < BN; ++j) a += (double)p[j] * vtd[((size_t)h * D + d) * N + j0 + j];
          oe[d] = oe[d] * alpha + (float)a;
        }
      }
      for (uint32_t d = 0; d < D; ++d) o_emu[((size_t)h * N + i) * D + d] = oe[d] / le;
    }

  // ---- device ----
  Runtime rt(kernel_file);
  fa_args_t fa = {};
#if E2E_TCU
  // Q and K go row-major for the DXA tile copies ([rows][DW] words); V^T and
  // the scale tensors keep their blocked layouts, whose per-block pieces are
  // runs of contiguous 32-word scale tiles.
  const uint32_t DW = D / F::epw, NKT_D = DW / KT_WORDS, nb = N / BN;
  auto row_major = [&](const QTensor& t) {
    std::vector<uint32_t> rm(t.words.size());
    for (uint32_t r = 0; r < t.rows; ++r)
      for (uint32_t w = 0; w < DW; ++w) rm[(size_t)r * DW + w] = t.words[blk_word(r, w, 16, NKT_D)];
    return rm;
  };
  fa.q = rt.upload(row_major(qq)); fa.k = rt.upload(row_major(kq)); fa.vt = rt.upload(vq.words);
  if (IS_MX) { fa.q_sc = rt.upload(qq.scales); fa.k_sc = rt.upload(kq.scales); fa.vt_sc = rt.upload(vq.scales); }
  using vortex::dxa::program_2d;
  const uint32_t LDQ = DW + 4, LDV = KT_WORDS + 4;
  E2E_CHECK(program_2d(rt.dev, 0, fa.q, DW, H * N, DW * 4, LDQ, BM, 4));
  E2E_CHECK(program_2d(rt.dev, 1, fa.k, DW, H * N, DW * 4, LDQ, BN, 4));
  E2E_CHECK(program_2d(rt.dev, 2, fa.vt, KT_WORDS, H * nb * D, KT_WORDS * 4, LDV, D, 4));
  if (IS_MX) {
    const uint32_t qk_tiles = H * N / 16 * NKT_D;
    E2E_CHECK(program_2d(rt.dev, 3, fa.q_sc, 32, qk_tiles, 128, 32, BM / 16 * NKT_D, 4));
    E2E_CHECK(program_2d(rt.dev, 4, fa.k_sc, 32, qk_tiles, 128, 32, BN / 16 * NKT_D, 4));
    E2E_CHECK(program_2d(rt.dev, 5, fa.vt_sc, 32, H * nb * (D / 16), 128, 32, D / 16, 4));
  }
  const uint32_t block = 1024, lmem = 65536;
#else
  fa.q = rt.upload(q); fa.k = rt.upload(k); fa.vt = rt.upload(v);
  const uint32_t block = 1024, lmem = 65536;
#endif
  fa.o = rt.alloc((uint64_t)H * N * D * 4);
  fa.ts_q = ts_q; fa.ts_k = ts_k; fa.ts_v = ts_v; fa.ts_p = ts_p; fa.qk_scale = qk_scale;
  fa.H = H; fa.N = N;
  rt.launch2("flash_attn", rt.kernel("flash_attn"), fa, N / BM, H, block, lmem);

  std::vector<float> o((size_t)H * N * D);
  rt.read(o.data(), fa.o, o.size() * 4);
  double e_emu = rel_rmse(o.data(), o_emu.data(), o.size());
  double e_ref = rel_rmse(o.data(), o_ref.data(), o.size());
  rt.report(e_emu, e_ref, e_emu < 2e-2);
  return e_emu < 2e-2 ? 0 : 1;
}
