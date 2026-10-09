#include "e2e_linear.h"
#include "common.h"

using namespace e2e;

__kernel void linear(linear_args_t* __UNIFORM__ arg) { linear_body(arg); }
__kernel void linear_swap(linear_args_t* __UNIFORM__ arg) { linear_swap_body(arg); }

E2E_INL float fsqrt_dev(float x) {
  float r;
  __asm__("fsqrt.s %0, %1" : "=f"(r) : "f"(x));
  return r;
}

// RMSNorm, one warp per row: dst = src * rsqrt(mean(src^2) + eps) * gamma.
__kernel void rmsnorm(llama_args_t* __UNIFORM__ arg) {
  const float* x = reinterpret_cast<const float*>(arg->src);
  const float* g = reinterpret_cast<const float*>(arg->gamma);
  float* y = reinterpret_cast<float*>(arg->dst);
  const uint32_t lane = threadIdx.x % NT, nw = gridDim.x * blockDim.x / NT;
  for (uint32_t r = (blockIdx.x * blockDim.x + threadIdx.x) / NT; r < arg->T; r += nw) {
    const float* xr = x + r * LL_DIM;
    float ss = 0.0f;
    for (uint32_t k = lane; k < LL_DIM; k += NT) ss = fma_dev(xr[k], xr[k], ss);
    ss = warp_sum(ss);
    const float inv = 1.0f / fsqrt_dev(ss / (float)LL_DIM + 1e-5f);
    for (uint32_t k = lane; k < LL_DIM; k += NT) y[r * LL_DIM + k] = xr[k] * inv * g[k];
  }
}

// fp32 [T][K] rows -> quantized activation operand (blocked by arb rows);
// one thread per run of QRUN elements.
__kernel void quant_act(llama_args_t* __UNIFORM__ arg) {
  const float* x = reinterpret_cast<const float*>(arg->src);
  const uint32_t K = arg->K, runs = K / QRUN, nkt = K / KT;
  for (uint32_t g = blockIdx.x * blockDim.x + threadIdx.x; g < arg->T * runs; g += gridDim.x * blockDim.x) {
    const uint32_t r = g / runs, c0 = (g % runs) * QRUN;
    quant_store_run(x + r * K + c0, QRUN, arg->ts_out,
                    reinterpret_cast<uint32_t*>(arg->act) + blk_word(r, c0 / F::epw, arg->arb, nkt),
                    IS_MX ? reinterpret_cast<uint8_t*>(arg->act_sc) + blk_sc(r, c0, arg->arb, nkt) : nullptr);
  }
}

// RoPE on q and k (adjacent-pair rotation, llama3-scaled frequencies) and the
// KV-cache append. One thread per (row, pair); v rides along with k.
__kernel void rope_kv(llama_args_t* __UNIFORM__ arg) {
  const float* qkv = reinterpret_cast<const float*>(arg->qkv);
  const float* freq = reinterpret_cast<const float*>(arg->freq);
  const int32_t* pos = reinterpret_cast<const int32_t*>(arg->pos);
  const int32_t* seq = reinterpret_cast<const int32_t*>(arg->seq);
  float* q = reinterpret_cast<float*>(arg->q);
  float* kc = reinterpret_cast<float*>(arg->kcache);
  float* vc = reinterpret_cast<float*>(arg->vcache);
  constexpr uint32_t PAIRS = (LL_DIM + LL_KV_DIM) / 2;
  for (uint32_t g = blockIdx.x * blockDim.x + threadIdx.x; g < arg->T * PAIRS; g += gridDim.x * blockDim.x) {
    const uint32_t t = g / PAIRS, p = g % PAIRS;
    const uint32_t col = 2 * p;
    const float ang = (float)pos[t] * freq[(col % LL_HEAD) / 2];
    const float c = cos_poly(ang), s = sin_poly(ang);
    const float* row = qkv + t * LL_QKV;
    const float a = row[col], b = row[col + 1];
    const float ra = a * c - b * s, rb = a * s + b * c;
    if (col < LL_DIM) {
      q[t * LL_DIM + col] = ra;
      q[t * LL_DIM + col + 1] = rb;
    } else {
      const uint32_t kcol = col - LL_DIM;
      const uint32_t off = ((uint32_t)seq[t] * arg->max_ctx + (uint32_t)pos[t]) * LL_KV_DIM + kcol;
      // K is cached transposed, [seq][kv_dim][ctx], so attention reads it key-contiguous.
      const uint32_t koff = ((uint32_t)seq[t] * LL_KV_DIM + kcol) * arg->max_ctx + (uint32_t)pos[t];
      kc[koff] = ra;
      kc[koff + arg->max_ctx] = rb;
      vc[off] = row[LL_DIM + LL_KV_DIM + kcol];
      vc[off + 1] = row[LL_DIM + LL_KV_DIM + kcol + 1];
    }
  }
}

// GQA attention, one warp per (row, query head) over the row's context
// (positions 0..pos): scores (lanes over keys, K^T cache), softmax, weighted
// sum of V (lanes over head dims). fp32 SIMT; no per-key warp reduction.
__kernel void attention(llama_args_t* __UNIFORM__ arg) {
  const float* q = reinterpret_cast<const float*>(arg->q);
  const float* kc = reinterpret_cast<const float*>(arg->kcache);
  const float* vc = reinterpret_cast<const float*>(arg->vcache);
  const int32_t* pos = reinterpret_cast<const int32_t*>(arg->pos);
  const int32_t* seq = reinterpret_cast<const int32_t*>(arg->seq);
  float* att = reinterpret_cast<float*>(arg->att);
  float* ao = reinterpret_cast<float*>(arg->ao);
  const uint32_t lane = threadIdx.x % NT, nw = gridDim.x * blockDim.x / NT, mc = arg->max_ctx;
  for (uint32_t w = (blockIdx.x * blockDim.x + threadIdx.x) / NT; w < arg->T * LL_HEADS; w += nw) {
    const uint32_t t = w / LL_HEADS, h = w % LL_HEADS, kvh = h / (LL_HEADS / LL_KV_HEADS);
    const uint32_t n = (uint32_t)pos[t] + 1;
    const float* qh = q + t * LL_DIM + h * LL_HEAD;
    const float* kt = kc + ((uint32_t)seq[t] * LL_KV_DIM + kvh * LL_HEAD) * mc;
    const float* vb = vc + (uint32_t)seq[t] * mc * LL_KV_DIM + kvh * LL_HEAD;
    float* a = att + w * mc;
    float mx = -1e30f;
    for (uint32_t j = lane; j < n; j += NT) {
      float s = 0.0f;
      for (uint32_t d = 0; d < LL_HEAD; ++d) s = fma_dev(qh[d], kt[d * mc + j], s);
      s *= arg->attn_scale;
      a[j] = s;
      mx = fmax_(mx, s);
    }
    mx = warp_max(mx);
    float sum = 0.0f;
    for (uint32_t j = lane; j < n; j += NT) {
      const float p = exp2_shared(a[j] - mx);
      a[j] = p;
      sum += p;
    }
    sum = warp_sum(sum);
    float o0 = 0.0f, o1 = 0.0f;
    for (uint32_t j = 0; j < n; ++j) {
      const float p = a[j];
      const float* vj = vb + j * LL_KV_DIM;
      o0 = fma_dev(p, vj[lane], o0);
      o1 = fma_dev(p, vj[lane + 32], o1);
    }
    const float inv = 1.0f / sum;
    ao[t * LL_DIM + h * LL_HEAD + lane] = o0 * inv;
    ao[t * LL_DIM + h * LL_HEAD + lane + 32] = o1 * inv;
  }
}
