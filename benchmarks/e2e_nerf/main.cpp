#include <unistd.h>
#include <algorithm>
#include "e2e_host.h"
#include "e2e_linear.h"
#include "common.h"

using namespace e2e;

static uint32_t R = 64, S = 32;
static const char* kernel_file = "kernel.vxbin";
static const float kNear = 0.05f, kFar = 6.0f;

#if E2E_TCU
static const uint32_t WRB = 64;    // weight row blocking = CTA tile N
#else
static const uint32_t WRB = 128;
#endif

static float pe_feature(const float* v, uint32_t f, uint32_t nf) {
  if (f < 3) return v[f];
  f -= 3;
  if (f >= 6 * nf) return 0.0f;
  const uint32_t l = f / 6, rem = f % 6, axis = rem / 2;
  const float x = v[axis] * (float)(1u << l);
  return (rem & 1) ? cos_poly(x) : sin_poly(x);
}

static float amax(const std::vector<float>& v) {
  float m = 0;
  for (float x : v) m = std::max(m, std::fabs(x));
  return m;
}

struct Layer {
  uint32_t n, k;
  std::vector<float> w, b;
  float ts = 1.0f;
  QTensor q;
};

// y[P][n] = act(x[P][k] . w^T + b)
static void lin(std::vector<float>& y, const std::vector<float>& x, const std::vector<float>& w,
                const std::vector<float>& b, uint32_t P, uint32_t k, uint32_t n, bool relu) {
  y.assign((size_t)P * n, 0.0f);
  for (uint32_t p = 0; p < P; ++p)
    for (uint32_t o = 0; o < n; ++o) {
      double s = 0;
      const float* xr = &x[(size_t)p * k];
      const float* wr = &w[(size_t)o * k];
      for (uint32_t i = 0; i < k; ++i) s += (double)xr[i] * wr[i];
      float t = (float)s + b[o];
      y[(size_t)p * n + o] = (relu && t < 0) ? 0.0f : t;
    }
}

static std::vector<float> concat(const std::vector<float>& a, uint32_t ka, const std::vector<float>& b,
                                 uint32_t kb, uint32_t P) {
  std::vector<float> o((size_t)P * (ka + kb));
  for (uint32_t p = 0; p < P; ++p) {
    std::copy(&a[(size_t)p * ka], &a[(size_t)p * ka] + ka, &o[(size_t)p * (ka + kb)]);
    std::copy(&b[(size_t)p * kb], &b[(size_t)p * kb] + kb, &o[(size_t)p * (ka + kb) + ka]);
  }
  return o;
}

int main(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "r:s:k:")) != -1) {
    if (c == 'r') R = atoi(optarg);
    if (c == 's') S = atoi(optarg);
    if (c == 'k') kernel_file = optarg;
  }
  const uint32_t P = R * S;
  if (P % LIN_BM) { std::printf("rays*samples must be a multiple of %u\n", LIN_BM); return -1; }
  const uint32_t kq = std::max(KT, 32u);
  const uint32_t k0 = round_up(3 + 6 * NERF_PE_X, kq), kd = round_up(3 + 6 * NERF_PE_D, kq);
  const uint32_t k5 = NERF_W + k0, k8 = NERF_W + kd;

  // ---- scene ----
  Rng rng(11);
  std::vector<float> ro(R * 3), rd(R * 3);
  for (uint32_t r = 0; r < R; ++r) {
    float v[3] = {rng.next(), rng.next(), rng.next()};
    float n = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) + 1e-6f;
    float d[3];
    for (int a = 0; a < 3; ++a) { ro[r * 3 + a] = 3.0f * v[a] / n; d[a] = -ro[r * 3 + a] / 3.0f + 0.3f * rng.next(); }
    n = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    for (int a = 0; a < 3; ++a) rd[r * 3 + a] = d[a] / n;
  }

  // ---- network: L0..L7 (skip into L5), feature layer, view layer ----
  std::vector<Layer> L(10);
  uint32_t kin[10] = {k0, NERF_W, NERF_W, NERF_W, NERF_W, k5, NERF_W, NERF_W, NERF_W, k8};
  uint32_t nout[10] = {NERF_W, NERF_W, NERF_W, NERF_W, NERF_W, NERF_W, NERF_W, NERF_W, NERF_W, NERF_WV};
  for (int l = 0; l < 10; ++l) {
    L[l].n = nout[l];
    L[l].k = kin[l];
    L[l].w.resize((size_t)L[l].n * L[l].k);
    L[l].b.resize(L[l].n);
    float a = std::sqrt(6.0f / L[l].k);
    for (auto& v : L[l].w) v = rng.next() * a;
    for (auto& v : L[l].b) v = rng.next() * 0.05f;
  }
  std::vector<float> w9(3 * NERF_WV), b9(3);
  for (auto& v : w9) v = rng.next() * std::sqrt(6.0f / NERF_WV);
  for (auto& v : b9) v = rng.next() * 0.05f;

  // ---- sample points and encodings (host mirror of ray_setup) ----
  std::vector<float> pex((size_t)P * k0, 0.0f), ped((size_t)P * kd, 0.0f), dl(P);
  for (uint32_t pt = 0; pt < P; ++pt) {
    uint32_t ray = pt / S, s = pt % S;
    const float* o = &ro[ray * 3];
    const float* d = &rd[ray * 3];
    float tmin = kNear, tmax = kFar;
    for (int a = 0; a < 3; ++a) {
      float inv = 1.0f / d[a];
      float t1 = (-1.0f - o[a]) * inv, t2 = (1.0f - o[a]) * inv;
      tmin = std::max(tmin, std::min(t1, t2));
      tmax = std::min(tmax, std::max(t1, t2));
    }
    float step = tmax > tmin ? (tmax - tmin) / (float)S : 0.0f;
    float t = tmin + ((float)s + 0.5f) * step;
    float p[3] = {o[0] + t * d[0], o[1] + t * d[1], o[2] + t * d[2]};
    dl[pt] = step;
    for (uint32_t f = 0; f < k0; ++f) pex[(size_t)pt * k0 + f] = pe_feature(p, f, NERF_PE_X);
    for (uint32_t f = 0; f < kd; ++f) ped[(size_t)pt * kd + f] = pe_feature(d, f, NERF_PE_D);
  }

  for (int l = 0; l < 10; ++l) {
    L[l].ts = tensor_scale_for(amax(L[l].w));
    L[l].q = quantize_host(L[l].w.data(), L[l].n, L[l].k, L[l].ts, WRB);
  }

  // ---- reference MLP; `emulate` quantizes every operand like the device ----
  float ts_in[10] = {};
  auto run_mlp = [&](bool emulate, std::vector<float>& h8, std::vector<float>& sig) {
    auto q = [&](const std::vector<float>& x, uint32_t k, int l) {
      if (!emulate) { ts_in[l] = std::max(ts_in[l], amax(x)); return x; }
      return quantize_host(x.data(), P, k, tensor_scale_for(ts_in[l]), 16).deq;
    };
    auto W = [&](int l) -> const std::vector<float>& { return emulate ? L[l].q.deq : L[l].w; };
    std::vector<float> h, t;
    lin(h, q(pex, k0, 0), W(0), L[0].b, P, k0, NERF_W, true);
    for (int l = 1; l <= 4; ++l) { lin(t, q(h, NERF_W, l), W(l), L[l].b, P, NERF_W, NERF_W, true); h.swap(t); }
    lin(t, q(concat(h, NERF_W, pex, k0, P), k5, 5), W(5), L[5].b, P, k5, NERF_W, true); h.swap(t);
    for (int l = 6; l <= 7; ++l) { lin(t, q(h, NERF_W, l), W(l), L[l].b, P, NERF_W, NERF_W, true); h.swap(t); }
    lin(t, q(h, NERF_W, 8), W(8), L[8].b, P, NERF_W, NERF_W, false); h.swap(t);
    sig.resize(P);
    for (uint32_t p = 0; p < P; ++p) sig[p] = h[(size_t)p * NERF_W];
    lin(h8, q(concat(h, NERF_W, ped, kd, P), k8, 9), W(9), L[9].b, P, k8, NERF_WV, true);
  };
  auto composite = [&](const std::vector<float>& h8, const std::vector<float>& sig) {
    std::vector<float> img(R * 3);
    for (uint32_t ray = 0; ray < R; ++ray) {
      float T = 1.0f, col[3] = {0, 0, 0};
      for (uint32_t s = 0; s < S; ++s) {
        uint32_t pt = ray * S + s;
        float l[3] = {b9[0], b9[1], b9[2]};
        for (uint32_t k = 0; k < NERF_WV; ++k)
          for (int ch = 0; ch < 3; ++ch) l[ch] += h8[(size_t)pt * NERF_WV + k] * w9[ch * NERF_WV + k];
        float sr = sig[pt];
        float sigma = sr > 15.0f ? sr : std::log1p(std::exp(sr));
        float alpha = 1.0f - std::exp(-sigma * dl[pt]);
        float w = alpha * T;
        for (int ch = 0; ch < 3; ++ch) col[ch] += w / (1.0f + std::exp(-l[ch]));
        T *= 1.0f - alpha;
      }
      for (int ch = 0; ch < 3; ++ch) img[ray * 3 + ch] = col[ch];
    }
    return img;
  };
  std::vector<float> h8_ref, sig_ref, h8_emu, sig_emu;
  run_mlp(false, h8_ref, sig_ref);
  std::vector<float> img_ref = composite(h8_ref, sig_ref), img_emu = img_ref;
#if E2E_TCU
  run_mlp(true, h8_emu, sig_emu);
  img_emu = composite(h8_emu, sig_emu);
#endif
  // concatenated operands share one tensor scale across their segments
  ts_in[5] = std::max(ts_in[5], ts_in[0]);

  double flops = 0;
  for (int l = 0; l < 10; ++l) flops += 2.0 * P * L[l].n * L[l].k;
  std::printf("E2E_APP app=nerf rays=%u samples=%u points=%u k0=%u kd=%u mlp_gemm_flops=%.0f\n", R, S, P, k0, kd, flops);

  // ---- device ----
  Runtime rt(kernel_file);
  auto k_lin = rt.kernel("linear"), k_setup = rt.kernel("ray_setup"), k_comp = rt.kernel("composite");
  auto k_head = rt.kernel("point_head");
  const uint32_t EPW = F::epw;
  auto opnd = [&](uint32_t k, uint64_t& sc) {
    uint64_t d = rt.alloc((uint64_t)P * k / EPW * 4);
    sc = IS_MX ? rt.alloc((uint64_t)(P / 16) * (k / KT) * SC_TILE_BYTES) : 0;
    return d;
  };
  nerf_args_t na = {};
  na.rays_o = rt.upload(ro); na.rays_d = rt.upload(rd);
  na.x0 = opnd(k0, na.x0_sc); na.x5 = opnd(k5, na.x5_sc); na.x8 = opnd(k8, na.x8_sc);
  uint64_t ha_sc, hb_sc;
  uint64_t ha = opnd(NERF_W, ha_sc), hb = opnd(NERF_W, hb_sc);
  na.deltas = rt.alloc(P * 4); na.sigma_raw = rt.alloc(P * 4); na.h8 = rt.alloc((uint64_t)P * NERF_WV * 4);
  na.w9 = rt.upload(w9); na.b9 = rt.upload(b9); na.image = rt.alloc(R * 3 * 4); na.rgba = rt.alloc((uint64_t)P * 16);
  na.ts_x0 = tensor_scale_for(ts_in[0]); na.ts_x5 = tensor_scale_for(ts_in[5]); na.ts_x8 = tensor_scale_for(ts_in[9]);
  na.n_rays = R; na.n_samples = S; na.n_points = P; na.k0 = k0; na.kd = kd; na.near = kNear; na.far = kFar;

  uint64_t wd[10], wsc[10], bd[10];
  for (int l = 0; l < 10; ++l) {
    wd[l] = rt.upload(L[l].q.words);
    wsc[l] = IS_MX ? rt.upload(L[l].q.scales) : 0;
    bd[l] = rt.upload(L[l].b);
  }

  const uint32_t fill_block = 256, fill_grid = VX_CFG_NUM_WARPS * VX_CFG_NUM_THREADS / fill_block;
#if E2E_TCU
  const uint32_t lmem = tcu_gemm_lmem(LIN_BM, LIN_BN);
#else
  const uint32_t lmem = simt_gemm_smem(LIN_BM, LIN_BN, 2);
#endif
  rt.launch1("ray_setup_pe", k_setup, na, fill_grid, fill_block);

  // layer l: input operand (a, a_sc, K), output (out, out_sc, ldo, col0, mode)
  auto layer = [&](int l, uint64_t a, uint64_t a_sc, float ts_a, uint64_t out, uint64_t out_sc, uint32_t ldo,
                   uint32_t col0, uint32_t mode, uint32_t act, float ts_out, uint64_t aux) {
    linear_args_t la = {};
    la.a = a; la.a_sc = a_sc; la.w = wd[l]; la.w_sc = wsc[l]; la.bias = bd[l];
    la.out = out; la.out_sc = out_sc; la.aux = aux;
    la.ts_a = ts_a; la.ts_w = L[l].ts; la.ts_out = ts_out;
    la.M = P; la.N = L[l].n; la.K = L[l].k; la.num_kt = L[l].k / KT;
    la.act = act; la.out_mode = mode; la.ldo = ldo; la.col0 = col0;
    rt.launch2("mlp_gemm", k_lin, la, L[l].n / LIN_BN, P / LIN_BM, 1024, lmem);
  };
  auto ts = [&](int l) { return tensor_scale_for(ts_in[l]); };
  layer(0, na.x0, na.x0_sc, ts(0), ha, ha_sc, NERF_W, 0, OUT_QUANT, ACT_RELU, ts(1), 0);
  layer(1, ha, ha_sc, ts(1), hb, hb_sc, NERF_W, 0, OUT_QUANT, ACT_RELU, ts(2), 0);
  layer(2, hb, hb_sc, ts(2), ha, ha_sc, NERF_W, 0, OUT_QUANT, ACT_RELU, ts(3), 0);
  layer(3, ha, ha_sc, ts(3), hb, hb_sc, NERF_W, 0, OUT_QUANT, ACT_RELU, ts(4), 0);
  layer(4, hb, hb_sc, ts(4), na.x5, na.x5_sc, k5, 0, OUT_QUANT, ACT_RELU, ts(5), 0);
  layer(5, na.x5, na.x5_sc, ts(5), ha, ha_sc, NERF_W, 0, OUT_QUANT, ACT_RELU, ts(6), 0);
  layer(6, ha, ha_sc, ts(6), hb, hb_sc, NERF_W, 0, OUT_QUANT, ACT_RELU, ts(7), 0);
  layer(7, hb, hb_sc, ts(7), ha, ha_sc, NERF_W, 0, OUT_QUANT, ACT_RELU, ts(8), 0);
  layer(8, ha, ha_sc, ts(8), na.x8, na.x8_sc, k8, 0, OUT_QUANT, ACT_NONE, ts(9), na.sigma_raw);
  layer(9, na.x8, na.x8_sc, ts(9), na.h8, 0, NERF_WV, 0, OUT_FP32, ACT_RELU, 1.0f, 0);
  rt.launch1("point_head", k_head, na, fill_grid, fill_block);
  rt.launch1("composite", k_comp, na, (R + fill_block - 1) / fill_block, fill_block);

  std::vector<float> img(R * 3);
  rt.read(img.data(), na.image, img.size() * 4);
  double e_emu = rel_rmse(img.data(), img_emu.data(), img.size());
  double e_ref = rel_rmse(img.data(), img_ref.data(), img.size());
  rt.report(e_emu, e_ref, e_emu < 2e-2);
  return e_emu < 2e-2 ? 0 : 1;
}
