// akili MoE: Mixtral-style top-2 gated expert FFN (SwiGLU).
//
// The router, token permutation, SwiGLU and combine run on the host. Each
// expert's two GEMMs run on the device through vx_matmul, with the expert
// weights as the A operand so the sparse modes can prune them 2:4:
//   H  [2*ffn x n_e] = W13_e [2*ffn x d] . X_e^T [d x n_e]
//   Y  [d x n_e]     = W2_e  [d x ffn]   . G     [ffn x n_e]
// Device cycles come from the AKILI_PERF line printed at exit.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

void vx_init();
void vx_matmul(float* C, float* A, float* B, int M, int N, int K);
void matmul_cleanup();

static constexpr int kTopK = 2;

static uint64_t g_rng = 88172645463325252ull;
static float urand() {
  g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
  return (float)((g_rng >> 40) & 0xFFFFFF) / 8388608.0f - 1.0f;
}

static void fill(std::vector<float>& v, float scale) {
  for (auto& x : v) x = urand() * scale;
}

// Keep the two largest magnitudes of every group of four along a row.
static void prune_2to4(std::vector<float>& w, size_t rows, size_t cols) {
  for (size_t r = 0; r < rows; ++r) {
    for (size_t c = 0; c + 4 <= cols; c += 4) {
      float* g = &w[r * cols + c];
      int idx[4] = {0, 1, 2, 3};
      std::sort(idx, idx + 4, [&](int a, int b) { return std::fabs(g[a]) > std::fabs(g[b]); });
      g[idx[2]] = 0.0f;
      g[idx[3]] = 0.0f;
    }
  }
}

// Round to the nearest fp16 value (ties to even), returned as float.
static float round_fp16(float f) {
  uint32_t b;
  std::memcpy(&b, &f, 4);
  const uint32_t sign = b & 0x80000000u;
  const int32_t e = (int32_t)((b >> 23) & 0xFF) - 127;
  if (e > 15) return f > 0 ? 65504.0f : -65504.0f;
  float a = std::fabs(f);
  const float q = std::ldexp(1.0f, (e < -14 ? -14 : e) - 10);
  float r = std::nearbyint(a / q) * q;
  return sign ? -r : r;
}

static inline float silu(float a) { return a / (1.0f + std::exp(-a)); }

struct Routing {
  std::vector<std::vector<std::pair<int, float>>> per_expert;  // (token, gate weight)
};

static Routing route(const std::vector<float>& x, const std::vector<float>& wg,
                     int T, int d, int E) {
  Routing r;
  r.per_expert.resize(E);
  std::vector<float> l(E);
  for (int t = 0; t < T; ++t) {
    float m = -1e30f;
    for (int e = 0; e < E; ++e) {
      float s = 0.0f;
      for (int k = 0; k < d; ++k) s += x[(size_t)t * d + k] * wg[(size_t)e * d + k];
      l[e] = s;
      m = std::max(m, s);
    }
    float sum = 0.0f;
    for (int e = 0; e < E; ++e) { l[e] = std::exp(l[e] - m); sum += l[e]; }
    int id[kTopK];
    float w[kTopK], wsum = 0.0f;
    for (int k = 0; k < kTopK; ++k) {
      int best = 0;
      for (int e = 1; e < E; ++e) if (l[e] > l[best]) best = e;
      id[k] = best;
      w[k] = l[best] / sum;
      wsum += w[k];
      l[best] = -1.0f;
    }
    for (int k = 0; k < kTopK; ++k) r.per_expert[id[k]].push_back({t, w[k] / wsum});
  }
  return r;
}

// One expert on the host (reference path).
static void expert_ref(const float* W13, const float* W2, const float* xt,
                       int d, int ffn, float w, float* y) {
  std::vector<float> h(2 * (size_t)ffn);
  for (int i = 0; i < 2 * ffn; ++i) {
    const float* row = W13 + (size_t)i * d;
    float s = 0.0f;
    for (int k = 0; k < d; ++k) s += row[k] * xt[k];
    h[i] = s;
  }
  std::vector<float> g(ffn);
  for (int i = 0; i < ffn; ++i) g[i] = silu(h[i]) * h[ffn + i];
  for (int c = 0; c < d; ++c) {
    const float* row = W2 + (size_t)c * ffn;
    float s = 0.0f;
    for (int k = 0; k < ffn; ++k) s += row[k] * g[k];
    y[c] += w * s;
  }
}

static bool parse_opt(char** argv, int& i, char opt, long& out) {
  const char* a = argv[i];
  if (a[0] != '-' || a[1] != opt) return false;
  out = a[2] ? atol(a + 2) : atol(argv[++i]);
  return true;
}

int main(int argc, char** argv) {
  long T = 64, d = 256, ffn = 256, E = 4, seed = 1, check = 1;
  for (int i = 1; i < argc; ++i) {
    if (parse_opt(argv, i, 't', T) || parse_opt(argv, i, 'd', d) ||
        parse_opt(argv, i, 'f', ffn) || parse_opt(argv, i, 'e', E) ||
        parse_opt(argv, i, 's', seed) || parse_opt(argv, i, 'c', check)) continue;
    fprintf(stderr, "unknown option %s\n", argv[i]);
    return 2;
  }
  if (d % 4 || ffn % 4) { fprintf(stderr, "d and ffn must be multiples of 4\n"); return 2; }
  g_rng ^= (uint64_t)seed * 0x9E3779B97F4A7C15ull;

#ifdef MOE_SPARSE
  const char* mode = "sparse";
#else
  const char* mode = "dense";
#endif
  printf("akili_moe (%s): T=%ld d=%ld ffn=%ld E=%ld topk=%d\n", mode, T, d, ffn, E, kTopK);

  std::vector<float> x((size_t)T * d), wg((size_t)E * d);
  std::vector<float> W13((size_t)E * 2 * ffn * d), W2((size_t)E * d * ffn);
  fill(x, 1.0f);
  fill(wg, 3.0f / std::sqrt((float)d));
  fill(W13, 1.0f / std::sqrt((float)d));
  fill(W2, 1.0f / std::sqrt((float)ffn));

  Routing r = route(x, wg, (int)T, (int)d, (int)E);
  size_t nmin = (size_t)-1, nmax = 0;
  for (auto& v : r.per_expert) { nmin = std::min(nmin, v.size()); nmax = std::max(nmax, v.size()); }
  printf("tokens per expert: min=%zu max=%zu\n", nmin, nmax);

  std::vector<float> y((size_t)T * d, 0.0f);
  vx_init();
  for (int e = 0; e < E; ++e) {
    const auto& as = r.per_expert[e];
    const int n = (int)as.size();
    if (n == 0) continue;
    float* w13 = W13.data() + (size_t)e * 2 * ffn * d;
    float* w2 = W2.data() + (size_t)e * d * ffn;
    std::vector<float> xt((size_t)d * n), h((size_t)2 * ffn * n), g((size_t)ffn * n), o((size_t)d * n);
    for (int j = 0; j < n; ++j)
      for (int k = 0; k < d; ++k) xt[(size_t)k * n + j] = x[(size_t)as[j].first * d + k];
    vx_matmul(h.data(), w13, xt.data(), (int)(2 * ffn), n, (int)d);
    for (int i = 0; i < ffn; ++i)
      for (int j = 0; j < n; ++j)
        g[(size_t)i * n + j] = silu(h[(size_t)i * n + j]) * h[(size_t)(ffn + i) * n + j];
    vx_matmul(o.data(), w2, g.data(), (int)d, n, (int)ffn);
    for (int j = 0; j < n; ++j) {
      const int tok = as[j].first;
      const float w = as[j].second;
      for (int c = 0; c < d; ++c) y[(size_t)tok * d + c] += w * o[(size_t)c * n + j];
    }
  }
  matmul_cleanup();

  if (!check) { printf("check skipped\n"); return 0; }

  auto reference = [&](std::vector<float> w13, std::vector<float> w2, bool round_first) {
    if (round_first) {
      for (auto& v : w13) v = round_fp16(v);
      for (auto& v : w2) v = round_fp16(v);
    }
#ifdef MOE_SPARSE
    prune_2to4(w13, (size_t)E * 2 * ffn, (size_t)d);
    prune_2to4(w2, (size_t)E * d, (size_t)ffn);
#endif
    std::vector<float> out((size_t)T * d, 0.0f), xt(d);
    for (int e = 0; e < E; ++e) {
      for (const auto& a : r.per_expert[e]) {
        for (int k = 0; k < d; ++k) xt[k] = x[(size_t)a.first * d + k];
        expert_ref(w13.data() + (size_t)e * 2 * ffn * d, w2.data() + (size_t)e * d * ffn,
                   xt.data(), (int)d, (int)ffn, a.second, &out[(size_t)a.first * d]);
      }
    }
    return out;
  };
  auto rel_err = [](const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
      double df = (double)a[i] - b[i];
      num += df * df;
      den += (double)b[i] * b[i];
    }
    return std::sqrt(num / (den > 0 ? den : 1.0));
  };
  // The device converts the weights to fp16 before pruning, so that is the
  // reference it is checked against; the fp32-prune difference is informational.
  std::vector<float> ref = reference(W13, W2, true);
  printf("fp32-prune reference differs from the fp16-round-then-prune reference by %.5f\n",
         rel_err(reference(W13, W2, false), ref));
  double num = 0.0, den = 0.0, maxd = 0.0;
  for (size_t i = 0; i < y.size(); ++i) {
    double df = (double)y[i] - ref[i];
    num += df * df;
    den += (double)ref[i] * ref[i];
    maxd = std::max(maxd, std::fabs(df));
  }
  const double rel = std::sqrt(num / (den > 0 ? den : 1.0));
  printf("rel L2 error vs fp16-rounded %s reference: %.5f (max abs %.5f)\n", mode, rel, maxd);
  if (rel > 0.03) { printf("FAILED!\n"); return 1; }
  printf("PASSED!\n");
  return 0;
}
