// GEMM-only harness: isolates staging, MMA and full-pipeline cost of the
// shared tcu_gemm / simt_gemm mainloops.
#include "e2e_dev.h"
#include "common.h"

using namespace e2e;

#if E2E_TCU
__kernel void g_full(gemm_args_t* __UNIFORM__ arg) {
  using G = tcu_gemm<GEMM_BM, GEMM_BN>;
  constexpr uint32_t CP = GEMM_BN + 1;
  auto smem = reinterpret_cast<uint32_t*>(__local_mem());
  float* cs = reinterpret_cast<float*>(smem);
  const uint32_t m0 = blockIdx.y * GEMM_BM, n0 = blockIdx.x * GEMM_BN;
  operand_t A{reinterpret_cast<const uint32_t*>(arg->a), reinterpret_cast<const uint32_t*>(arg->a_sc), m0, 0, 0};
  operand_t B{reinterpret_cast<const uint32_t*>(arg->b), reinterpret_cast<const uint32_t*>(arg->b_sc), n0, 0, 0};
  G::run(smem, A, B, arg->num_kt, cs, CP);
  float* c = reinterpret_cast<float*>(arg->c);
  for (uint32_t i = threadIdx.x; i < GEMM_BM * GEMM_BN; i += blockDim.x) {
    uint32_t r = i / GEMM_BN, col = i % GEMM_BN;
    c[(m0 + r) * arg->N + n0 + col] = cs[r * CP + col];
  }
}
#if E2E_TCU
// Microbenchmarks of the MX metadata path (no staging, no fragment loads).
// g_tculd: 2 TCU_LDs per iteration; g_tculd_mma: + one WMMA on register data.
template <bool MMA>
static inline __attribute__((always_inline)) void tculd_body(gemm_args_t* arg) {
  auto smem = reinterpret_cast<uint32_t*>(__local_mem());
  const uint32_t wid = get_sub_group_id();
  ctx::fragment_a fa;
  ctx::fragment_b fb;
  ctx::fragment_acc acc;
  ctx::fill_fragment(fa, 0);
  ctx::fill_fragment(fb, 0);
  ctx::fill_fragment(acc, 0);
  for (uint32_t i = 0; i < arg->num_kt; ++i) {
    if constexpr (IS_MX) {
      ctx::load_mx_metadata(fa, smem + (wid % 8) * 32);
      ctx::load_mx_metadata(fb, smem + 256 + (wid % 4) * 32);
    }
    if constexpr (MMA) ctx::mma_sync(acc, fa, fb, acc);
  }
  auto pc = reinterpret_cast<float*>(arg->c) + (blockIdx.y * GEMM_BM + (wid / 4) * 16) * arg->N + blockIdx.x * GEMM_BN + (wid % 4) * 16;
  ctx::store_matrix_sync(pc, acc, arg->N);
}
__kernel void g_tculd(gemm_args_t* __UNIFORM__ arg) { tculd_body<false>(arg); }
__kernel void g_tculd_mma(gemm_args_t* __UNIFORM__ arg) { tculd_body<true>(arg); }
#endif
#else
__kernel void g_full(gemm_args_t* __UNIFORM__ arg) {
  using S = simt_gemm<128, 128>;
  auto smem = reinterpret_cast<float*>(__local_mem());
  const uint32_t m0 = blockIdx.y * 128, n0 = blockIdx.x * 128;
  float acc[4][4] = {};
  operand_t A{reinterpret_cast<const uint32_t*>(arg->a), nullptr, m0, 1, 0};
  operand_t B{reinterpret_cast<const uint32_t*>(arg->b), nullptr, n0, 0, 0};
  S::run(acc, smem, A, B, arg->num_kt);
  float* c = reinterpret_cast<float*>(arg->c);
  for (uint32_t i = 0; i < 4; ++i)
    for (uint32_t j = 0; j < 4; ++j)
      c[(m0 + S::ty() + 32 * i) * arg->N + n0 + S::tx() + 32 * j] = acc[i][j];
}
#endif
