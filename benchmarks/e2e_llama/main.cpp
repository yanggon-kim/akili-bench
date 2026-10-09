#include <unistd.h>
#include <algorithm>
#include <string>
#include "e2e_host.h"
#include "e2e_linear.h"
#include "common.h"

using namespace e2e;

// prefill: T prompt tokens of one sequence, causal attention over them.
// decode : T sequences, one new token each, attending over ctx cached tokens.
static std::string mode = "decode";
static uint32_t T = 32, CTX = 512;
static const char* kernel_file = "kernel.vxbin";

static float amax(const std::vector<float>& v) {
  float m = 0;
  for (float x : v) m = std::max(m, std::fabs(x));
  return m;
}

// Llama-3 RoPE frequency scaling (rope_theta 500000, factor 32, low 1, high 4, orig 8192).
static std::vector<float> rope_freqs() {
  const double theta = 500000.0, factor = 32.0, low = 1.0, high = 4.0, orig = 8192.0;
  std::vector<float> f(LL_HEAD / 2);
  for (uint32_t i = 0; i < LL_HEAD / 2; ++i) {
    double fr = std::pow(theta, -2.0 * i / LL_HEAD);
    double wavelen = 2.0 * M_PI / fr;
    if (wavelen > orig / low) {
      fr = fr / factor;
    } else if (wavelen >= orig / high) {
      double smooth = (orig / wavelen - low) / (high - low);
      fr = (1.0 - smooth) * fr / factor + smooth * fr;
    }
    f[i] = (float)fr;
  }
  return f;
}

// y[T][N] = x[T][K] . w[N][K]^T
static void gemm(std::vector<float>& y, const std::vector<float>& x, const std::vector<float>& w,
                 uint32_t rows, uint32_t K, uint32_t N) {
  y.assign((size_t)rows * N, 0.0f);
  for (uint32_t t = 0; t < rows; ++t)
    for (uint32_t o = 0; o < N; ++o) {
      double s = 0;
      const float* xr = &x[(size_t)t * K];
      const float* wr = &w[(size_t)o * K];
      for (uint32_t k = 0; k < K; ++k) s += (double)xr[k] * wr[k];
      y[(size_t)t * N + o] = (float)s;
    }
}

int main(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "m:b:t:l:k:")) != -1) {
    if (c == 'm') mode = optarg;
    if (c == 'b' || c == 't') T = atoi(optarg);
    if (c == 'l') CTX = atoi(optarg);
    if (c == 'k') kernel_file = optarg;
  }
  const bool decode = (mode == "decode");
  const uint32_t ARB = decode ? LSW_BN : LIN_BM;          // activation row blocking
  const uint32_t WRB = decode ? LSW_BM : LIN_BN;          // weight row blocking
  if (T % ARB) { std::printf("rows must be a multiple of %u\n", ARB); return -1; }
  const uint32_t seqs = decode ? T : 1;
  const uint32_t max_ctx = decode ? CTX + 1 : T;
  const float attn_scale = 1.44269504f / std::sqrt((float)LL_HEAD);
  const double gemm_flops = 2.0 * T * ((double)LL_QKV * LL_DIM + (double)LL_DIM * LL_DIM + 3.0 * LL_FFN * LL_DIM);
  std::printf("E2E_APP app=llama_%s model=llama3.2-1b layers_run=1 dim=%u heads=%u kv_heads=%u head_dim=%u ffn=%u "
              "rows=%u context=%u gemm_flops=%.0f\n", mode.c_str(), LL_DIM, LL_HEADS, LL_KV_HEADS, LL_HEAD, LL_FFN,
              T, decode ? CTX + 1 : T, gemm_flops);

  // ---- layer weights ----
  Rng rng(17);
  auto init = [&](uint32_t n, uint32_t k) {
    std::vector<float> w((size_t)n * k);
    const float a = std::sqrt(3.0f / k);
    for (auto& v : w) v = rng.next() * a;
    return w;
  };
  std::vector<float> wqkv = init(LL_QKV, LL_DIM), wo = init(LL_DIM, LL_DIM);
  std::vector<float> w1 = init(LL_FFN, LL_DIM), w3 = init(LL_FFN, LL_DIM), w2 = init(LL_DIM, LL_FFN);
  std::vector<float> g1(LL_DIM), g2(LL_DIM);
  for (auto& v : g1) v = 1.0f + 0.1f * rng.next();
  for (auto& v : g2) v = 1.0f + 0.1f * rng.next();
  std::vector<float> w13((size_t)2 * LL_FFN * LL_DIM);   // gate/up interleaved in SWIGLU_G groups
  for (uint32_t j = 0; j < LL_FFN; ++j) {
    uint32_t g = (j / SWIGLU_G) * 2 * SWIGLU_G + j % SWIGLU_G;
    std::copy(&w1[(size_t)j * LL_DIM], &w1[(size_t)j * LL_DIM] + LL_DIM, &w13[(size_t)g * LL_DIM]);
    std::copy(&w3[(size_t)j * LL_DIM], &w3[(size_t)j * LL_DIM] + LL_DIM, &w13[(size_t)(g + SWIGLU_G) * LL_DIM]);
  }
  const std::vector<float> freq = rope_freqs();

  // ---- inputs: residual stream, positions, KV history ----
  std::vector<float> x((size_t)T * LL_DIM);
  for (auto& v : x) v = rng.next();
  std::vector<int32_t> pos(T), seq(T);
  for (uint32_t t = 0; t < T; ++t) { pos[t] = decode ? (int32_t)CTX : (int32_t)t; seq[t] = decode ? (int32_t)t : 0; }
  std::vector<float> kc0((size_t)seqs * max_ctx * LL_KV_DIM, 0.0f), vc0 = kc0;
  if (decode) {
    for (uint32_t s = 0; s < seqs; ++s)
      for (uint32_t p = 0; p < CTX; ++p)
        for (uint32_t d = 0; d < LL_KV_DIM; ++d) {
          kc0[((size_t)s * max_ctx + p) * LL_KV_DIM + d] = rng.next();
          vc0[((size_t)s * max_ctx + p) * LL_KV_DIM + d] = rng.next();
        }
  }

  // ---- weights as the device sees them ----
  const float ts_qkv = tensor_scale_for(amax(wqkv)), ts_o = tensor_scale_for(amax(wo));
  const float ts_13 = tensor_scale_for(amax(w13)), ts_2 = tensor_scale_for(amax(w2));
  QTensor qqkv = quantize_host(wqkv.data(), LL_QKV, LL_DIM, ts_qkv, WRB);
  QTensor qo = quantize_host(wo.data(), LL_DIM, LL_DIM, ts_o, WRB);
  QTensor q13 = quantize_host(w13.data(), 2 * LL_FFN, LL_DIM, ts_13, WRB);
  QTensor q2 = quantize_host(w2.data(), LL_DIM, LL_FFN, ts_2, WRB);

  // ---- reference layer; `emulate` quantizes like the device ----
  float ts_act[4] = {};   // calibrated amax: rmsnorm1, attention out, rmsnorm2, swiglu out
  auto layer = [&](bool emulate) {
    auto Q = [&](std::vector<float> v, uint32_t K, int i) {
      if (!emulate) { ts_act[i] = std::max(ts_act[i], amax(v)); return v; }
      for (uint32_t r = 0; r < T; ++r) quant_dequant_row(&v[(size_t)r * K], K, tensor_scale_for(ts_act[i]), &v[(size_t)r * K]);
      return v;
    };
    auto W = [&](const std::vector<float>& w, const QTensor& q) -> const std::vector<float>& { return emulate ? q.deq : w; };
    auto rms = [&](const std::vector<float>& in, const std::vector<float>& g) {
      std::vector<float> o(in.size());
      for (uint32_t r = 0; r < T; ++r) {
        double ss = 0;
        for (uint32_t k = 0; k < LL_DIM; ++k) ss += (double)in[(size_t)r * LL_DIM + k] * in[(size_t)r * LL_DIM + k];
        float inv = 1.0f / std::sqrt((float)(ss / LL_DIM) + 1e-5f);
        for (uint32_t k = 0; k < LL_DIM; ++k) o[(size_t)r * LL_DIM + k] = in[(size_t)r * LL_DIM + k] * inv * g[k];
      }
      return o;
    };
    std::vector<float> qkv, kc = kc0, vc = vc0, ao((size_t)T * LL_DIM), o, x1, gu, h((size_t)T * LL_FFN), dn, y;
    gemm(qkv, Q(rms(x, g1), LL_DIM, 0), W(wqkv, qqkv), T, LL_DIM, LL_QKV);
    std::vector<float> qr((size_t)T * LL_DIM);
    for (uint32_t t = 0; t < T; ++t)
      for (uint32_t col = 0; col < LL_DIM + LL_KV_DIM; col += 2) {
        const float ang = (float)pos[t] * freq[(col % LL_HEAD) / 2];
        const float cc = cos_poly(ang), ss = sin_poly(ang);
        const float a = qkv[(size_t)t * LL_QKV + col], b = qkv[(size_t)t * LL_QKV + col + 1];
        const float ra = a * cc - b * ss, rb = a * ss + b * cc;
        if (col < LL_DIM) { qr[(size_t)t * LL_DIM + col] = ra; qr[(size_t)t * LL_DIM + col + 1] = rb; }
        else {
          const uint32_t kcol = col - LL_DIM;
          const size_t off = ((size_t)seq[t] * max_ctx + pos[t]) * LL_KV_DIM + kcol;
          kc[off] = ra; kc[off + 1] = rb;
          vc[off] = qkv[(size_t)t * LL_QKV + LL_DIM + LL_KV_DIM + kcol];
          vc[off + 1] = qkv[(size_t)t * LL_QKV + LL_DIM + LL_KV_DIM + kcol + 1];
        }
      }
    std::vector<float> sc(max_ctx);
    for (uint32_t t = 0; t < T; ++t)
      for (uint32_t hh = 0; hh < LL_HEADS; ++hh) {
        const uint32_t kvh = hh / (LL_HEADS / LL_KV_HEADS), n = pos[t] + 1;
        float mx = -1e30f;
        for (uint32_t j = 0; j < n; ++j) {
          double s = 0;
          for (uint32_t d = 0; d < LL_HEAD; ++d)
            s += (double)qr[(size_t)t * LL_DIM + hh * LL_HEAD + d] * kc[((size_t)seq[t] * max_ctx + j) * LL_KV_DIM + kvh * LL_HEAD + d];
          sc[j] = (float)s * attn_scale;
          mx = std::max(mx, sc[j]);
        }
        double sum = 0;
        for (uint32_t j = 0; j < n; ++j) { sc[j] = exp2_shared(sc[j] - mx); sum += sc[j]; }
        for (uint32_t d = 0; d < LL_HEAD; ++d) {
          double a = 0;
          for (uint32_t j = 0; j < n; ++j) a += (double)sc[j] * vc[((size_t)seq[t] * max_ctx + j) * LL_KV_DIM + kvh * LL_HEAD + d];
          ao[(size_t)t * LL_DIM + hh * LL_HEAD + d] = (float)(a / sum);
        }
      }
    gemm(o, Q(ao, LL_DIM, 1), W(wo, qo), T, LL_DIM, LL_DIM);
    x1 = x;
    for (size_t i = 0; i < x1.size(); ++i) x1[i] += o[i];
    gemm(gu, Q(rms(x1, g2), LL_DIM, 2), W(w13, q13), T, LL_DIM, 2 * LL_FFN);
    for (uint32_t t = 0; t < T; ++t)
      for (uint32_t j = 0; j < LL_FFN; ++j) {
        uint32_t g = (j / SWIGLU_G) * 2 * SWIGLU_G + j % SWIGLU_G;
        float gv = gu[(size_t)t * 2 * LL_FFN + g], uv = gu[(size_t)t * 2 * LL_FFN + g + SWIGLU_G];
        h[(size_t)t * LL_FFN + j] = gv / (1.0f + std::exp(-gv)) * uv;
      }
    gemm(dn, Q(h, LL_FFN, 3), W(w2, q2), T, LL_FFN, LL_DIM);
    y = x1;
    for (size_t i = 0; i < y.size(); ++i) y[i] += dn[i];
    return y;
  };
  std::vector<float> y_ref = layer(false);
#if E2E_TCU
  std::vector<float> y_emu = layer(true);
#else
  std::vector<float> y_emu = y_ref;
#endif

  // ---- device ----
  Runtime rt(kernel_file);
  auto k_lin = rt.kernel(decode ? "linear_swap" : "linear");
  auto k_rms = rt.kernel("rmsnorm"), k_q = rt.kernel("quant_act");
  auto k_rope = rt.kernel("rope_kv"), k_att = rt.kernel("attention");
  llama_args_t la = {};
  la.x = rt.upload(x);
  uint64_t d_g1 = rt.upload(g1), d_g2 = rt.upload(g2);
  la.dst = rt.alloc((uint64_t)T * LL_DIM * 4);
  uint64_t act_sc = 0, h_sc = 0;
  uint64_t act = rt.alloc((uint64_t)T * LL_DIM / F::epw * 4);
  uint64_t hact = rt.alloc((uint64_t)T * LL_FFN / F::epw * 4);
  if (IS_MX) {
    act_sc = rt.alloc((uint64_t)(T / 16) * (LL_DIM / KT) * SC_TILE_BYTES);
    h_sc = rt.alloc((uint64_t)(T / 16) * (LL_FFN / KT) * SC_TILE_BYTES);
  }
  la.qkv = rt.alloc((uint64_t)T * LL_QKV * 4);
  la.q = rt.alloc((uint64_t)T * LL_DIM * 4);
  std::vector<float> kct(kc0.size());   // device K cache is [seq][kv_dim][ctx]
  for (uint32_t sq = 0; sq < seqs; ++sq)
    for (uint32_t p = 0; p < max_ctx; ++p)
      for (uint32_t d = 0; d < LL_KV_DIM; ++d)
        kct[((size_t)sq * LL_KV_DIM + d) * max_ctx + p] = kc0[((size_t)sq * max_ctx + p) * LL_KV_DIM + d];
  la.kcache = rt.upload(kct); la.vcache = rt.upload(vc0);
  la.att = rt.alloc((uint64_t)T * LL_HEADS * max_ctx * 4);
  la.ao = rt.alloc((uint64_t)T * LL_DIM * 4);
  la.freq = rt.upload(freq); la.pos = rt.upload(pos); la.seq = rt.upload(seq);
  la.attn_scale = attn_scale; la.T = T; la.arb = ARB; la.max_ctx = max_ctx;
  uint64_t d_wqkv = rt.upload(qqkv.words), d_wo = rt.upload(qo.words), d_w13 = rt.upload(q13.words), d_w2 = rt.upload(q2.words);
  uint64_t s_wqkv = 0, s_wo = 0, s_w13 = 0, s_w2 = 0;
  if (IS_MX) { s_wqkv = rt.upload(qqkv.scales); s_wo = rt.upload(qo.scales); s_w13 = rt.upload(q13.scales); s_w2 = rt.upload(q2.scales); }

  const uint32_t fill_block = 256, fill_grid = VX_CFG_NUM_WARPS * VX_CFG_NUM_THREADS / fill_block;
  const uint32_t lmem = decode ? lin_swap_lmem() : lin_lmem();
  auto quant = [&](uint64_t src, uint32_t K, uint64_t dst, uint64_t dsc, int i) {
    la.src = src; la.K = K; la.act = dst; la.act_sc = dsc; la.ts_out = tensor_scale_for(ts_act[i]);
    rt.launch1("quant_act", k_q, la, fill_grid, fill_block);
  };
  auto gemm_dev = [&](const char* name, uint64_t w, uint64_t wsc, float ts_w, uint64_t a, uint64_t asc, int ia,
                      uint32_t N, uint32_t K, uint32_t act_fn, uint32_t omode, uint64_t out, uint64_t osc,
                      uint32_t ldo, int iout, uint64_t resid) {
    linear_args_t l = {};
    l.a = a; l.a_sc = asc; l.w = w; l.w_sc = wsc; l.out = out; l.out_sc = osc; l.resid = resid;
    l.ts_a = tensor_scale_for(ts_act[ia]); l.ts_w = ts_w; l.ts_out = iout >= 0 ? tensor_scale_for(ts_act[iout]) : 1.0f;
    l.M = T; l.N = N; l.K = K; l.num_kt = K / KT; l.act = act_fn; l.out_mode = omode; l.ldo = ldo;
    if (decode) rt.launch2(name, k_lin, l, N / LSW_BM, T / LSW_BN, 1024, lmem);
    else        rt.launch2(name, k_lin, l, N / LIN_BN, T / LIN_BM, 1024, lmem);
  };

  // attention block
  la.src = la.x; la.gamma = d_g1;
  rt.launch1("rmsnorm", k_rms, la, fill_grid, fill_block);
  quant(la.dst, LL_DIM, act, act_sc, 0);
  gemm_dev("qkv_gemm", d_wqkv, s_wqkv, ts_qkv, act, act_sc, 0, LL_QKV, LL_DIM, ACT_NONE, OUT_FP32, la.qkv, 0, LL_QKV, -1, 0);
  rt.launch1("rope_kv", k_rope, la, fill_grid, fill_block);
  rt.launch1("attention", k_att, la, fill_grid, fill_block);
  quant(la.ao, LL_DIM, act, act_sc, 1);
  gemm_dev("o_gemm_resid", d_wo, s_wo, ts_o, act, act_sc, 1, LL_DIM, LL_DIM, ACT_NONE, OUT_FP32, la.x, 0, LL_DIM, -1, la.x);
  // FFN block
  la.src = la.x; la.gamma = d_g2;
  rt.launch1("rmsnorm", k_rms, la, fill_grid, fill_block);
  quant(la.dst, LL_DIM, act, act_sc, 2);
  gemm_dev("ffn_w13_swiglu", d_w13, s_w13, ts_13, act, act_sc, 2, 2 * LL_FFN, LL_DIM, ACT_SWIGLU, OUT_QUANT, hact, h_sc,
           LL_FFN, 3, 0);
  gemm_dev("ffn_down_resid", d_w2, s_w2, ts_2, hact, h_sc, 3, LL_DIM, LL_FFN, ACT_NONE, OUT_FP32, la.x, 0, LL_DIM, -1, la.x);

  // Errors on the layer's update y - x (the residual input would dominate y).
  std::vector<float> y((size_t)T * LL_DIM);
  rt.read(y.data(), la.x, y.size() * 4);
  for (size_t i = 0; i < y.size(); ++i) { y[i] -= x[i]; y_emu[i] -= x[i]; y_ref[i] -= x[i]; }
  double e_emu = rel_rmse(y.data(), y_emu.data(), y.size());
  double e_ref = rel_rmse(y.data(), y_ref.data(), y.size());
  rt.report(e_emu, e_ref, e_emu < 2e-2);
  return e_emu < 2e-2 ? 0 : 1;
}
