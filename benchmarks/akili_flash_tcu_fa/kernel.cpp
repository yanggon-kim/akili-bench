// akili_flash_tcu_fa — fused FlashAttention-2 forward on WGMMA + DXA.
//
// One CTA owns FA_BR query rows (4 warps x xtileM). Q is fetched once into
// LMEM; K/V tiles of FA_BC keys stream through a 2-stage DXA pipeline. Per
// tile: S = Q Kᵀ (WGMMA), online softmax per query row (one lane per row),
// rescale the O accumulator, O += P V (WGMMA). The N x N score matrix never
// leaves the core.
//
// LMEM (16 KB at d = 32): Q 2 KB | 2 x (K 2 KB + Vᵀ 2 KB) | P 2 KB | S 4 KB.
// Each row's softmax rescale factor (and finally 1/l) is published in that
// row's first S slot, which the row's lane has already consumed.

#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_intrinsics.h>
#include <vx_dxa.h>
#include <vx_barrier.h>
#include "common.h"

namespace vt = vortex::tensor;
using ctx = vt::wgmma_context<VX_CFG_NUM_THREADS, vt::fp16, vt::fp32, false, 32>;

constexpr uint32_t kDescQ = 0;
constexpr uint32_t kDescK = 1;
constexpr uint32_t kDescV = 2;

static_assert(ctx::xtileN == FA_BC, "K/V tile width must equal xtileN");
static_assert(ctx::xtileN == FA_D, "P·V output width must equal xtileN");
static_assert(FA_BR % ctx::xtileM == 0 && ctx::xtileM == VX_CFG_NUM_THREADS,
              "one lane per query row within a warp");
static_assert(FA_D % ctx::tileK == 0 && FA_BC % ctx::tileK == 0, "K multiple of tileK");

constexpr uint32_t kQBytes     = FA_BR * FA_D * 2;
constexpr uint32_t kKBytes     = FA_BC * FA_D * 2;
constexpr uint32_t kStageBytes = 2 * kKBytes;
constexpr uint32_t kKVOff      = kQBytes;
constexpr uint32_t kPOff       = kKVOff + 2 * kStageBytes;
constexpr uint32_t kSOff       = kPOff + FA_BR * FA_BC * 2;
constexpr uint32_t kSmemBytes  = kSOff + FA_BR * FA_BC * 4;
static_assert(kSmemBytes <= 16384, "FlashAttention tiles exceed the 16 KB LMEM");

// Fast fp32 exp: Taylor(x/32) then y^32, clamped. Same as akili_attn's softmax.
static inline float fast_exp(float x) {
  x = x < -80.0f ? -80.0f : x;
  float u = x * 0.03125f;
  float y = 1.0f + u*(1.0f + u*(0.5f + u*(0.166666667f + u*(0.041666667f + u*0.008333333f))));
  y = y*y; y = y*y; y = y*y; y = y*y; y = y*y;
  return y;
}

static inline uint16_t f2h(float x) {
  uint32_t bits;
  __builtin_memcpy(&bits, &x, sizeof(bits));
  uint16_t sign = (uint16_t)((bits >> 16) & 0x8000);
  uint16_t mant = (uint16_t)((bits >> 13) & 0x03FF);
  int32_t  ex   = (int32_t)((bits >> 23) & 0xFF) - 127 + 15;
  if (ex >= 31) return (uint16_t)(sign | 0x7C00);
  if (ex <= 0)  return sign;
  return (uint16_t)(sign | ((uint16_t)ex << 10) | mant);
}

// Row (within the warp's xtileM rows) held by accumulator register r of this
// lane; mirrors wgmma_context::store_matrix_sync's n-major register layout.
static inline uint32_t acc_row(uint32_t lane, uint32_t r) {
  return lane / ctx::tcN + (r % ctx::m_steps) * ctx::tcM;
}

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t N         = arg->N;
  const uint32_t lane      = threadIdx.x % VX_CFG_NUM_THREADS;
  const uint32_t warp_rank = threadIdx.x / VX_CFG_NUM_THREADS;
  const uint32_t num_warps = blockDim.x / VX_CFG_NUM_THREADS;
  const uint32_t row0      = blockIdx.x * FA_BR;
  const uint32_t num_tiles = N / FA_BC;

  auto smem = reinterpret_cast<uint8_t*>(__local_mem());
  auto Q = reinterpret_cast<ctx::input_t*>(smem);
  auto stage_K = [&](uint32_t s) { return reinterpret_cast<ctx::input_t*>(smem + kKVOff + s * kStageBytes); };
  auto stage_V = [&](uint32_t s) { return reinterpret_cast<ctx::input_t*>(smem + kKVOff + s * kStageBytes + kKBytes); };
  auto Pw = reinterpret_cast<uint16_t*>(smem + kPOff) + warp_rank * ctx::xtileM * FA_BC;
  auto Sw = reinterpret_cast<float*>(smem + kSOff) + warp_rank * ctx::xtileM * FA_BC;
  auto Qw = Q + warp_rank * ctx::xtileM * FA_D;

  const uint32_t cta = get_local_group_id();
  auto bar_kv = [&](uint32_t s) { return cta | (s << 8); };
  const uint32_t bar_q    = cta | (2u << 8);
  const uint32_t bar_sync = cta | (3u << 8);
  const bool dxa_warp = (get_sub_group_id() == 0);

  auto issue_kv = [&](uint32_t s, uint32_t tile) {
    vx_barrier_expect_tx(bar_kv(s), 2);
    vx_dxa_issue_2d_wg(kDescK, bar_kv(s), stage_K(s), 0, tile * FA_BC);
    vx_dxa_issue_2d_wg(kDescV, bar_kv(s), stage_V(s), 0, tile * FA_BC);
  };

  if (dxa_warp) {
    vx_barrier_expect_tx(bar_q, 1);
    vx_dxa_issue_2d_wg(kDescQ, bar_q, Q, 0, row0);
    issue_kv(0, 0);
  }
  vx_barrier(bar_q, num_warps);

  ctx::fragment_acc fragO;
  ctx::fill_fragment(fragO, 0);
  float m_i = -1.0e30f;
  float l_i = 0.0f;
  float* srow = Sw + lane * FA_BC;
  uint16_t* prow = Pw + lane * FA_BC;

  for (uint32_t t = 0; t < num_tiles; ++t) {
    const uint32_t cur = t & 1u;
    if (t + 1 < num_tiles && dxa_warp)
      issue_kv(cur ^ 1u, t + 1);
    vx_barrier(bar_kv(cur), num_warps);

    // S = Q Kᵀ over the head dimension.
    ctx::fragment_acc fragS;
    ctx::fill_fragment(fragS, 0);
    for (uint32_t k = 0; k < FA_D; k += ctx::tileK) {
      auto desc_a = vt::vx_make_smem_desc(Qw + k, FA_D * sizeof(ctx::input_t));
      auto desc_b = vt::vx_make_smem_desc(stage_K(cur) + k, FA_D * sizeof(ctx::input_t));
      ctx::wgmma_sync(fragS, desc_a, desc_b, fragS);
    }
    ctx::store_matrix_sync(Sw, fragS, FA_BC);
    vx_fence();
    vx_barrier(bar_sync, num_warps);

    // Online softmax: this lane owns query row (warp_rank * xtileM + lane).
    float m_new = m_i;
    for (uint32_t j = 0; j < FA_BC; ++j)
      m_new = srow[j] > m_new ? srow[j] : m_new;
    float scale = fast_exp(m_i - m_new);
    float sum = 0.0f;
    for (uint32_t j = 0; j < FA_BC; ++j) {
      float p = fast_exp(srow[j] - m_new);
      sum += p;
      prow[j] = f2h(p);
    }
    l_i = l_i * scale + sum;
    m_i = m_new;
    srow[0] = scale;
    vx_fence();
    vx_barrier(bar_sync, num_warps);

    for (uint32_t r = 0; r < ctx::fragment_acc::NR; ++r)
      fragO.data[r] *= Sw[acc_row(lane, r) * FA_BC];

    // O += P V over this tile's keys.
    for (uint32_t k = 0; k < FA_BC; k += ctx::tileK) {
      auto desc_a = vt::vx_make_smem_desc(Pw + k, FA_BC * sizeof(uint16_t));
      auto desc_b = vt::vx_make_smem_desc(stage_V(cur) + k, FA_BC * sizeof(ctx::input_t));
      ctx::wgmma_sync(fragO, desc_a, desc_b, fragO);
    }

    vx_barrier(bar_kv(cur), num_warps);
  }

  srow[0] = 1.0f / l_i;
  vx_fence();
  vx_barrier(bar_sync, num_warps);
  for (uint32_t r = 0; r < ctx::fragment_acc::NR; ++r)
    fragO.data[r] *= Sw[acc_row(lane, r) * FA_BC];

  auto pO = reinterpret_cast<ctx::output_t*>(arg->O_addr)
          + (row0 + warp_rank * ctx::xtileM) * FA_D;
  ctx::store_matrix_sync(pO, fragO, FA_D);
}
