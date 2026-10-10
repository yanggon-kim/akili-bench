#ifndef FA_SOFTMAX_H
#define FA_SOFTMAX_H

// Row softmax step for the fused FlashAttention kernels: one lane owns one
// query row of FA_BC scores. exp is evaluated as 2^n * poly(f) (degree 3,
// relative error 7.5e-5, below the fp16 resolution the probabilities are
// stored in), and the scores are processed FA_SM_U at a time in staged loops
// so the independent dependency chains overlap in the in-order pipeline.

#include <stdint.h>

#ifndef FA_SM_U
#ifdef FA_SPARSE
#define FA_SM_U 1
#else
#define FA_SM_U 2
#endif
#endif

static inline float fa_bits_to_float(uint32_t b) {
  float f;
  __builtin_memcpy(&f, &b, sizeof(f));
  return f;
}

static inline uint32_t fa_float_to_bits(float f) {
  uint32_t b;
  __builtin_memcpy(&b, &f, sizeof(b));
  return b;
}

// 2^y for y <= 0, flushed to the smallest normal exponent below -126.
static inline float fa_exp2_nonpos(float y) {
  y = __builtin_fmaxf(y, -126.0f);
  int n;
  __asm__("fcvt.w.s %0, %1, rdn" : "=r"(n) : "f"(y));
  const float f = y - (float)n;
  float p = 0.0780244739f;
  p = 0.226067225f + f * p;
  p = 0.695833517f + f * p;
  p = 0.99992522f + f * p;
  return p * fa_bits_to_float((uint32_t)(n + 127) << 23);
}

// fp16 bits of p in [0, 2): truncating, values below 2^-14 flush to zero.
static inline uint16_t fa_f2h_unit(float p) {
  const uint32_t b = fa_float_to_bits(p);
  const uint32_t h = (b >> 13) - (112u << 10);
  return (uint16_t)(h & (0u - (uint32_t)(b >= 0x38800000u)));
}

// Updates the running max m_i and sum l_i for this tile, writes the fp16
// probabilities to prow, and returns the factor to rescale the accumulator by.
static inline float fa_softmax_tile(const float* srow, uint16_t* prow,
                                    float& m_i, float& l_i) {
  constexpr float kLog2e = 1.4426950409f;

  float mx[FA_SM_U];
  mx[0] = m_i;
#pragma unroll
  for (uint32_t u = 1; u < FA_SM_U; ++u) mx[u] = -1.0e30f;
#pragma unroll
  for (uint32_t j = 0; j < FA_BC; j += FA_SM_U) {
#pragma unroll
    for (uint32_t u = 0; u < FA_SM_U; ++u)
      mx[u] = __builtin_fmaxf(mx[u], srow[j + u]);
  }
  float m_new = mx[0];
#pragma unroll
  for (uint32_t u = 1; u < FA_SM_U; ++u) m_new = __builtin_fmaxf(m_new, mx[u]);
  const float scale = fa_exp2_nonpos((m_i - m_new) * kLog2e);

  float sum[FA_SM_U];
#pragma unroll
  for (uint32_t u = 0; u < FA_SM_U; ++u) sum[u] = 0.0f;
#pragma unroll
  for (uint32_t j = 0; j < FA_BC; j += FA_SM_U) {
    float y[FA_SM_U], q[FA_SM_U], p[FA_SM_U];
    int n[FA_SM_U];
#pragma unroll
    for (uint32_t u = 0; u < FA_SM_U; ++u)
      y[u] = __builtin_fmaxf((srow[j + u] - m_new) * kLog2e, -126.0f);
#pragma unroll
    for (uint32_t u = 0; u < FA_SM_U; ++u)
      __asm__("fcvt.w.s %0, %1, rdn" : "=r"(n[u]) : "f"(y[u]));
#pragma unroll
    for (uint32_t u = 0; u < FA_SM_U; ++u)
      y[u] -= (float)n[u];
#pragma unroll
    for (uint32_t u = 0; u < FA_SM_U; ++u)
      q[u] = 0.226067225f + y[u] * 0.0780244739f;
#pragma unroll
    for (uint32_t u = 0; u < FA_SM_U; ++u)
      q[u] = 0.695833517f + y[u] * q[u];
#pragma unroll
    for (uint32_t u = 0; u < FA_SM_U; ++u)
      q[u] = 0.99992522f + y[u] * q[u];
#pragma unroll
    for (uint32_t u = 0; u < FA_SM_U; ++u)
      p[u] = q[u] * fa_bits_to_float((uint32_t)(n[u] + 127) << 23);
#pragma unroll
    for (uint32_t u = 0; u < FA_SM_U; ++u)
      sum[u] += p[u];
#pragma unroll
    for (uint32_t u = 0; u < FA_SM_U; ++u)
      prow[j + u] = fa_f2h_unit(p[u]);
  }
  float total = sum[0];
#pragma unroll
  for (uint32_t u = 1; u < FA_SM_U; ++u) total += sum[u];
  l_i = l_i * scale + total;
  m_i = m_new;
  return scale;
}

#endif
