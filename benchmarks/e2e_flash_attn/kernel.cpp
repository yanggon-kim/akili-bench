#include "e2e_dev.h"
#include "common.h"

using namespace e2e;

#if E2E_TCU
// ---------------------------------------------------------------------------
// FlashAttention-3 style forward on the TCU.
//   CTA = 48 query rows of one head. Warp-specialized: one warp on issue
//   lane 3 is the producer (DXA copies of Q and each K / V^T block); the 24
//   warps off lane 3 form 3 consumer groups of 8 warps x 16 rows.
//   Per KV block (BN = one TCU K-tile of keys): S = Q K^T on the TCU (warp t
//   of a group computes S tile t), online softmax (exp2, running max/sum),
//   P quantized on-device to the operand format (block scales for MX/NV, a
//   fixed tensor scale for fp8/nvfp4), then O += P V^T^T on the TCU with each
//   warp owning one 16-wide head-dim slice of O in registers.
// ---------------------------------------------------------------------------
using ctx = vt::wmma_context<NT, IT, vt::fp32>;
constexpr uint32_t DW = FA_D / F::epw;          // words per Q/K row
constexpr uint32_t NKT_D = DW / KT_WORDS;       // TCU K-tiles along the head dim
constexpr uint32_t BN = KT;                     // keys per block
constexpr uint32_t LDQ = DW + 4;                // padded smem row (conflict-free fragments)
constexpr uint32_t LDV = KT_WORDS + 4;
constexpr uint32_t ELEM = 4 / sizeof(ctx::input_t);
constexpr uint32_t SP = BN + 1;                 // S scratch pitch (floats)
constexpr uint32_t GROUPS = 3;
static_assert(FA_BM == GROUPS * 16, "3 groups of 16 query rows");

// smem layout (words). MX scale tiles live next to their operands so every
// TCU_LD hits smem (an L1 miss would hold the shared TCU for a DRAM trip).
constexpr uint32_t SCT = 32;                                // words per scale tile
constexpr uint32_t OFF_Q = 0;
constexpr uint32_t OFF_K = OFF_Q + FA_BM * LDQ;
constexpr uint32_t OFF_V = OFF_K + BN * LDQ;
constexpr uint32_t OFF_S = OFF_V + FA_D * LDV;
constexpr uint32_t OFF_P = OFF_S + GROUPS * 16 * SP;
constexpr uint32_t OFF_ST = OFF_P + GROUPS * 16 * LDV;      // m, l, alpha per row
constexpr uint32_t OFF_QSC = (OFF_ST + 3 * FA_BM + SCT - 1) / SCT * SCT;
constexpr uint32_t OFF_KSC = OFF_QSC + (IS_MX ? GROUPS * NKT_D * SCT : 0);
constexpr uint32_t OFF_VSC = OFF_KSC + (IS_MX ? BN / 16 * NKT_D * SCT : 0);
constexpr uint32_t OFF_PSC = OFF_VSC + (IS_MX ? FA_D / 16 * SCT : 0);
constexpr uint32_t SMEM_WORDS = OFF_PSC + (IS_MX ? GROUPS * SCT : 0);
static_assert(SMEM_WORDS * 4 <= 65536, "fits in 64 KB smem");

// DXA descriptor slots (programmed by the host).
constexpr uint32_t DS_Q = 0, DS_K = 1, DS_V = 2, DS_QSC = 3, DS_KSC = 4, DS_VSC = 5;
// CTA barriers: K tile full / consumed, V tile full / consumed (2..4 are the
// per-group consumer barriers).
constexpr uint32_t B_KF = 0, B_KE = 1, B_VF = 5, B_VE = 6;

// Producer: one warp streams Q (once) and each K / V^T block (plus MX scale
// tiles) into smem with DXA. K and V have separate full/consumed barriers so
// K(j+1) streams during softmax / P quantization and V(j+1) during the next
// S = Q K^T, hiding the DXA latency behind consumer work.
static __attribute__((noinline)) void fa_producer(const fa_args_t* arg, uint32_t* sm, uint32_t h, uint32_t r0,
                                                  bool issuer, uint32_t id0, uint32_t nw) {
  const stage_bars bars{id0, nw};
  const uint32_t nb = arg->N / BN;
  constexpr uint32_t NTX = IS_MX ? 2 : 1;
  auto issue_k = [&](uint32_t j) {
    vx_dxa_issue_2d_wg(DS_K, bars.id(B_KF), sm + OFF_K, 0, h * arg->N + j * BN);
    if constexpr (IS_MX)
      vx_dxa_issue_2d_wg(DS_KSC, bars.id(B_KF), sm + OFF_KSC, 0, (h * arg->N + j * BN) / 16 * NKT_D);
  };
  auto issue_v = [&](uint32_t j) {
    vx_dxa_issue_2d_wg(DS_V, bars.id(B_VF), sm + OFF_V, 0, (h * nb + j) * FA_D);
    if constexpr (IS_MX)
      vx_dxa_issue_2d_wg(DS_VSC, bars.id(B_VF), sm + OFF_VSC, 0, (h * nb + j) * (FA_D / 16));
  };
  if (issuer) {
    bars.expect(B_KF, 2 * NTX);
    vx_dxa_issue_2d_wg(DS_Q, bars.id(B_KF), sm + OFF_Q, 0, h * arg->N + r0);
    if constexpr (IS_MX)
      vx_dxa_issue_2d_wg(DS_QSC, bars.id(B_KF), sm + OFF_QSC, 0, (h * arg->N + r0) / 16 * NKT_D);
    issue_k(0);
    bars.expect(B_VF, NTX);
    issue_v(0);
  }
  for (uint32_t j = 0; j < nb; ++j) {
    const bool more = issuer && j + 1 < nb;
    bars.sync(B_KF);
    bars.sync(B_KE);
    if (more) { bars.expect(B_KF, NTX); issue_k(j + 1); }
    bars.sync(B_VF);
    bars.sync(B_VE);
    if (more) { bars.expect(B_VF, NTX); issue_v(j + 1); }
  }
}

// Quantizes the F::epw values at x into one operand word. MX/NV: the words
// of a scale block sit on adjacent lanes; their amax is reduced by butterfly
// and the block's first lane writes the scale byte.
E2E_INL uint32_t quant_word(const float* x, uint32_t w, float tscale, uint8_t* sc) {
  float v[F::epw];
  for (uint32_t e = 0; e < F::epw; ++e) v[e] = x[e];
  float mult = (F::scale == SCALE_TENSOR) ? 1.0f / tscale : 1.0f;
  if constexpr (IS_MX) {
    constexpr uint32_t WPB = F::blk / F::epw;
    float amax = 0.0f;
    for (uint32_t e = 0; e < F::epw; ++e) amax = fmax_(amax, fabs_(v[e]));
    for (uint32_t b = WPB / 2; b > 0; b >>= 1) amax = fmax_(amax, u2f((uint32_t)vx_shfl_bfly(f2u(amax), b, 15, 0x30)));
    uint32_t sf;
    if constexpr (F::scale == SCALE_MX) {
      sf = e8m0_select(amax, 1.0f / (F::bits == 8 ? E4M3_MAX : E2M1_MAX));
      mult = e8m0_inv(sf);
    } else {
      sf = nv_select(amax, 1.0f / (E2M1_MAX * tscale));
      mult = 1.0f / (e4m3_to_f32(sf) * tscale);
    }
    if (w % WPB == 0) *sc = (uint8_t)sf;
  }
  uint32_t word = 0;
  for (uint32_t e = 0; e < F::epw; ++e) word |= encode(v[e], mult) << (e * F::bits);
  return word;
}

static __attribute__((noinline)) void fa_consumer(const fa_args_t* arg, uint32_t* sm, uint32_t h, uint32_t r0,
                                                  uint32_t c, uint32_t id0, uint32_t nw) {
  const stage_bars bars{id0, nw};
  const uint32_t g = c / 8, gw = c % 8, lane = threadIdx.x % NT;
  const uint32_t gbar = ((2 + g) << 8) + get_local_group_id();
  float* S = reinterpret_cast<float*>(sm + OFF_S) + g * 16 * SP;
  uint32_t* P = sm + OFF_P + g * 16 * LDV;
  uint32_t* psc = sm + OFF_PSC + g * SCT;
  float* st_m = reinterpret_cast<float*>(sm + OFF_ST) + g * 16;
  float* st_l = st_m + FA_BM;
  float* st_a = st_l + FA_BM;
  const float qk_c = arg->ts_q * arg->ts_k * arg->qk_scale;
  // softmax: 16 threads per row, E contiguous keys each. P quantization: one
  // operand word per thread (a 16 x BN P tile is always 16 x 16 words).
  constexpr uint32_t E = BN / 16;
  const uint32_t tt = gw * NT + lane, srow = tt / 16, spart = tt % 16;
  if (spart == 0) { st_m[srow] = -1e30f; st_l[srow] = 0.0f; }

  ctx::fragment_acc oacc;
  ctx::fill_fragment(oacc, 0);
  const uint32_t nb = arg->N / BN;
  for (uint32_t j = 0; j < nb; ++j) {
    bars.sync(B_KF);
    // S tile gw = Q_g . K_gw^T
    if (gw < BN / 16) {
      ctx::fragment_a fa;
      ctx::fragment_b fb;
      ctx::fragment_acc sacc;
      ctx::fill_fragment(sacc, 0);
      for (uint32_t kt = 0; kt < NKT_D; ++kt) {
        if constexpr (IS_MX) {
          ctx::load_mx_metadata(fa, sm + OFF_QSC + (g * NKT_D + kt) * SCT);
          ctx::load_mx_metadata(fb, sm + OFF_KSC + (gw * NKT_D + kt) * SCT);
        }
        ctx::load_matrix_sync(fa, sm + OFF_Q + g * 16 * LDQ + kt * KT_WORDS, LDQ * ELEM);
        ctx::load_matrix_sync<vt::col_major>(fb, sm + OFF_K + gw * 16 * LDQ + kt * KT_WORDS, LDQ * ELEM);
        ctx::mma_sync(sacc, fa, fb, sacc);
      }
      ctx::store_matrix_sync(S + gw * 16, sacc, SP);
    }
    bars.sync(B_KE);
    // online softmax over this block's keys
    {
      float* row = S + srow * SP + spart * E;
      float mx = -1e30f;
      for (uint32_t e = 0; e < E; ++e) mx = fmax_(mx, row[e] * qk_c);
      for (int b = 8; b > 0; b >>= 1) mx = fmax_(mx, u2f((uint32_t)vx_shfl_bfly(f2u(mx), b, 15, 0x30)));
      const float m_old = st_m[srow];
      const float m_new = fmax_(m_old, mx);
      float sum = 0.0f;
      for (uint32_t e = 0; e < E; ++e) {
        const float p = exp2_shared(row[e] * qk_c - m_new);
        row[e] = p;
        sum += p;
      }
      for (int b = 8; b > 0; b >>= 1) sum += u2f((uint32_t)vx_shfl_bfly(f2u(sum), b, 15, 0x30));
      if (spart == 0) {
        const float alpha = exp2_shared(m_old - m_new);
        st_m[srow] = m_new;
        st_l[srow] = st_l[srow] * alpha + sum;
        st_a[srow] = alpha;
      }
    }
    vx_barrier(gbar, 8);
    // quantize P (16 x BN) into the A-operand tile
    {
      const uint32_t r = tt / KT_WORDS, w = tt % KT_WORDS;
      P[r * LDV + w] = quant_word(S + r * SP + w * F::epw, w, arg->ts_p,
                                  reinterpret_cast<uint8_t*>(psc) + r * SC_PER_ROW + w * F::epw / BLK);
    }
    vx_barrier(gbar, 8);
    bars.sync(B_VF);
    // O_gw = alpha * O_gw + P . V^T_gw
    {
      for (uint32_t r = 0; r < 8; ++r) oacc.data[r] *= st_a[lane / 4 + 8 * (r / 4)];
      ctx::fragment_a fa;
      ctx::fragment_b fb;
      if constexpr (IS_MX) {
        ctx::load_mx_metadata(fa, psc);
        ctx::load_mx_metadata(fb, sm + OFF_VSC + gw * SCT);
      }
      ctx::load_matrix_sync(fa, P, LDV * ELEM);
      ctx::load_matrix_sync<vt::col_major>(fb, sm + OFF_V + gw * 16 * LDV, LDV * ELEM);
      ctx::mma_sync(oacc, fa, fb, oacc);
    }
    bars.sync(B_VE);
  }
  // O = O / l
  const float os = arg->ts_p * arg->ts_v;
  float* o = reinterpret_cast<float*>(arg->o);
  for (uint32_t r = 0; r < 8; ++r) {
    const uint32_t row = lane / 4 + 8 * (r / 4), col = lane % 4 + 4 * (r % 4);
    o[(h * arg->N + r0 + g * 16 + row) * FA_D + gw * 16 + col] = oacc.data[r] * os / st_l[row];
  }
}

__kernel void flash_attn(fa_args_t* __UNIFORM__ arg) {
  auto sm = reinterpret_cast<uint32_t*>(__local_mem());
  const uint32_t h = blockIdx.y, r0 = blockIdx.x * FA_BM;
  const uint32_t wid = get_sub_group_id();
  const stage_bars bars = stage_bars::make(0);
  if ((wid % VX_CFG_ISSUE_WIDTH) == VX_CFG_ISSUE_WIDTH - 1) {
    fa_producer(arg, sm, h, r0, wid == VX_CFG_ISSUE_WIDTH - 1, bars.id0, bars.nw);
  } else {
    const uint32_t c = (wid / VX_CFG_ISSUE_WIDTH) * (VX_CFG_ISSUE_WIDTH - 1) + wid % VX_CFG_ISSUE_WIDTH;
    fa_consumer(arg, sm, h, r0, c, bars.id0, bars.nw);
  }
}

#else
// ---------------------------------------------------------------------------
// FPU baseline: SIMT FlashAttention (FA-2 style). CTA = 32 query rows x 1024
// threads, 32-key blocks staged in K-major smem. Lane mappings are diagonal
// (warp w, lane l) so every smem access of a warp hits distinct banks.
// S[i][j]: i = l, j = (l+w)%32. O[i][d]: i = l, d = (l+w)%32 + 32c (in regs).
// ---------------------------------------------------------------------------
constexpr uint32_t BM = FA_SIMT_BM, BN = FA_SIMT_BN, D = FA_D;
constexpr uint32_t PQ = BM + 1, PV = D + 1;
constexpr uint32_t OFF_Q = 0, OFF_K = OFF_Q + D * PQ, OFF_V = OFF_K + D * PQ, OFF_P = OFF_V + BN * PV;
constexpr uint32_t OFF_ST = OFF_P + BN * PQ;
constexpr uint32_t SMEM_WORDS = OFF_ST + 3 * BM;
static_assert(SMEM_WORDS * 4 <= 65536, "fits in 64 KB smem");

__kernel void flash_attn(fa_args_t* __UNIFORM__ arg) {
  float* sm = reinterpret_cast<float*>(__local_mem());
  const uint32_t h = blockIdx.y, r0 = blockIdx.x * BM, N = arg->N;
  const uint32_t tid = threadIdx.x, w = tid / 32, l = tid % 32;
  const uint32_t jj = (l + w) % 32;
  const float* q = reinterpret_cast<const float*>(arg->q) + (h * N + r0) * D;
  const float* kg = reinterpret_cast<const float*>(arg->k) + h * N * D;
  const float* vg = reinterpret_cast<const float*>(arg->vt) + h * N * D;
  float* st_m = sm + OFF_ST;
  float* st_l = st_m + BM;
  float* st_a = st_l + BM;
  for (uint32_t idx = tid; idx < BM * D; idx += 1024) sm[OFF_Q + (idx % D) * PQ + idx / D] = q[idx];
  if (tid < BM) { st_m[tid] = -1e30f; st_l[tid] = 0.0f; }
  float o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  const float qk = arg->qk_scale;
  for (uint32_t j0 = 0; j0 < N; j0 += BN) {
    for (uint32_t idx = tid; idx < BN * D; idx += 1024) {
      sm[OFF_K + (idx % D) * PQ + idx / D] = kg[j0 * D + idx];
      sm[OFF_V + (idx / D) * PV + idx % D] = vg[j0 * D + idx];
    }
    __syncthreads();
    float s = 0.0f;
    const float* qa = sm + OFF_Q + l;
    const float* ka = sm + OFF_K + jj;
    for (uint32_t k = 0; k < D; ++k) s = fma_dev(qa[k * PQ], ka[k * PQ], s);
    sm[OFF_P + jj * PQ + l] = s * qk;      // P^T[j][i]
    __syncthreads();
    // softmax: warp w owns row w, lane = key
    {
      const float v = sm[OFF_P + l * PQ + w];
      const float m_old = st_m[w];
      const float m_new = fmax_(m_old, warp_max(v));
      const float p = exp2_shared(v - m_new);
      const float sum = warp_sum(p);
      sm[OFF_P + l * PQ + w] = p;
      if (l == 0) {
        const float alpha = exp2_shared(m_old - m_new);
        st_m[w] = m_new;
        st_l[w] = st_l[w] * alpha + sum;
        st_a[w] = alpha;
      }
    }
    __syncthreads();
    const float a = st_a[l];
    for (uint32_t c = 0; c < 4; ++c) o[c] *= a;
    for (uint32_t j = 0; j < BN; ++j) {
      const float p = sm[OFF_P + j * PQ + l];
      const float* vr = sm + OFF_V + j * PV + jj;
      for (uint32_t c = 0; c < 4; ++c) o[c] = fma_dev(p, vr[32 * c], o[c]);
    }
    __syncthreads();
  }
  float* og = reinterpret_cast<float*>(arg->o) + (h * N + r0 + l) * D;
  const float inv = 1.0f / st_l[l];
  for (uint32_t c = 0; c < 4; ++c) og[jj + 32 * c] = o[c] * inv;
}
#endif
