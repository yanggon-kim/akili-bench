#ifndef AKILI_WGMMA_DXA_HOST_H
#define AKILI_WGMMA_DXA_HOST_H

#include <dxa.h>
#include <tensor_cfg.h>
#include <vortex2.h>
#include <vector>
#include <cstdio>
#include "wgmma_dxa_layout.h"

// Warps per WGMMA CTA (cta_m = warps * xtileM). Defaults to the issue width;
// AKILI_CTA_WARPS overrides it, e.g. to reuse each B tile across more rows.
inline uint32_t akili_cta_warps(uint64_t issue_width) {
#ifdef AKILI_CTA_WARPS
  (void)issue_width;
  return AKILI_CTA_WARPS;
#else
  return (uint32_t)issue_width;
#endif
}

// K elements one WGMMA covers: sparse FULLK doubles it (Hopper wgmma.sp k32).
template <typename WgCfg>
constexpr uint32_t akili_wg_tile_k(bool sparse) {
  return sparse ? WgCfg::tileK_sp : WgCfg::tileK;
}

// K elements per DEEP_BUF stage: the DXA descriptors' K tile size.
template <typename WgCfg>
constexpr uint32_t akili_wg_stage_k(bool sparse) {
  return akili::kWgKB * akili_wg_tile_k<WgCfg>(sparse);
}

// Launch lmem_size for the PIPE_N-stage pipeline (wgmma_dxa_kernel.h).
template <typename WgCfg>
constexpr uint32_t akili_wg_lmem_bytes(uint32_t cta_m, bool sparse) {
  return akili::kWgPipeN * (sparse
      ? akili::wg_sp_stage_bytes(cta_m, WgCfg::tileK_sp, WgCfg::xtileN, WgCfg::xtileM)
      : akili::wg_dense_stage_bytes(cta_m, WgCfg::tileK, WgCfg::xtileN));
}

template <typename WgCfg, typename InputType>
inline void akili_pack_wgmma_metadata(std::vector<uint32_t>& metadata,
                                      const std::vector<uint8_t>& masks,
                                      uint32_t M, uint32_t K) {
  constexpr uint32_t rtl_i_ratio = 32 / InputType::bits;
  constexpr uint32_t meta_row_bits = WgCfg::tcK * 2 * rtl_i_ratio;
  constexpr uint32_t words_per_tile = VX_CFG_NUM_THREADS;
  using WmCfg = vortex::tensor::wmma_config_t<VX_CFG_NUM_THREADS>;
  constexpr uint32_t pwd_wmma = WmCfg::m_steps * (WmCfg::k_steps / 2);
  constexpr uint32_t rtl_half_k_wmma = WmCfg::k_steps / 2;
  constexpr uint32_t dense_per_step = WgCfg::tcK * rtl_i_ratio * 2;

  uint32_t tile_rows = M / WgCfg::xtileM;
  constexpr uint32_t sub_steps = WgCfg::tileK_sp / dense_per_step;
  static_assert(sub_steps >= 1 && sub_steps <= rtl_half_k_wmma, "sparse sub-steps exceed metadata banks");
  uint32_t k_tiles = K / WgCfg::tileK_sp;
  uint32_t groups_per_row = K / 4;
  metadata.assign((size_t)tile_rows * k_tiles * words_per_tile, 0);

  for (uint32_t tr = 0; tr < tile_rows; ++tr) {
    for (uint32_t kt = 0; kt < k_tiles; ++kt) {
      uint32_t tile_base = (tr * k_tiles + kt) * words_per_tile;
      for (uint32_t sm = 0; sm < WgCfg::m_steps; ++sm) {
        for (uint32_t sk = 0; sk < sub_steps; ++sk) {
          uint32_t bank = sm * rtl_half_k_wmma + sk;
          uint32_t k_start = kt * WgCfg::tileK_sp + sk * dense_per_step;
          for (uint32_t row = 0; row < WgCfg::tcM; ++row) {
            uint32_t physical_row = tr * WgCfg::xtileM + sm * WgCfg::tcM + row;
            for (uint32_t group = 0; group < dense_per_step / 4; ++group) {
              uint8_t mask = masks[physical_row * groups_per_row
                                 + k_start / 4 + group];
              for (uint32_t pos = 0; pos < 4; ++pos) {
                if (!(mask & (1u << pos))) continue;
                uint32_t element = group * 4 + pos;
                uint32_t k_reg = element / (2 * rtl_i_ratio);
                uint32_t in_reg = element % (2 * rtl_i_ratio);
                uint32_t meta_bit = in_reg < rtl_i_ratio
                    ? k_reg * rtl_i_ratio + in_reg
                    : (WgCfg::tcK + k_reg) * rtl_i_ratio
                      + in_reg - rtl_i_ratio;
                uint32_t bit = row * meta_row_bits + meta_bit;
                uint32_t lane = bank + (bit / 32) * pwd_wmma;
                metadata[tile_base + lane] |= 1u << (bit % 32);
              }
            }
          }
        }
      }
    }
  }
}

// Device-side cost of every akili_start_g launch, from core 0's always-on
// MCYCLE/MINSTRET counters read around the launch; totals print at exit.
struct AkiliPerf {
  uint64_t launches = 0, cycles = 0, instrs = 0;
  ~AkiliPerf() {
    if (launches)
      printf("AKILI_PERF: launches=%llu cycles=%llu instrs=%llu\n",
             (unsigned long long)launches, (unsigned long long)cycles,
             (unsigned long long)instrs);
  }
};
inline AkiliPerf& akili_perf() { static AkiliPerf perf; return perf; }

inline int akili_read_core_counters(vx_device_h device, uint64_t* cycles,
                                    uint64_t* instrs) {
  if (int ret = vx_device_mpm_query(device, VX_DCR_MPM_CLASS_CORE, VX_CSR_MCYCLE, 0, cycles)) return ret;
  return vx_device_mpm_query(device, VX_DCR_MPM_CLASS_CORE, VX_CSR_MINSTRET, 0, instrs);
}

inline int akili_start_g(vx_device_h device, const void* args, size_t args_size,
                         uint32_t ndim, const uint32_t* grid_dim,
                         const uint32_t* block_dim, uint32_t lmem_size) {
  static vx_device_h cached_device = nullptr;
  static vx_module_h module = nullptr;
  static vx_kernel_h kernel = nullptr;
  static vx_queue_h queue = nullptr;
  if (cached_device != device) {
    if (queue) vx_queue_release(queue);
    if (kernel) vx_kernel_release(kernel);
    if (module) vx_module_release(module);
    cached_device = device;
    module = nullptr;
    kernel = nullptr;
    queue = nullptr;
    if (int ret = vx_module_load_file(device, "kernel.vxbin", &module)) return ret;
    if (int ret = vx_module_get_kernel(module, "main", &kernel)) return ret;
    if (int ret = vx_queue_create(device, nullptr, &queue)) return ret;
  }
  vx_launch_info_t info = {};
  info.struct_size = sizeof(info);
  info.kernel = kernel;
  info.args_host = args;
  info.args_size = args_size;
  info.ndim = ndim;
  info.grid_dim[0] = grid_dim[0];
  info.grid_dim[1] = ndim > 1 ? grid_dim[1] : 1;
  info.grid_dim[2] = ndim > 2 ? grid_dim[2] : 1;
  info.block_dim[0] = block_dim[0];
  info.block_dim[1] = ndim > 1 ? block_dim[1] : 1;
  info.block_dim[2] = ndim > 2 ? block_dim[2] : 1;
  info.lmem_size = lmem_size;
  uint64_t c0 = 0, i0 = 0, c1 = 0, i1 = 0;
  if (int ret = akili_read_core_counters(device, &c0, &i0)) return ret;
  int ret = vx_enqueue_launch(queue, &info, 0, nullptr, nullptr);
  if (ret) return ret;
  if ((ret = vx_queue_finish(queue, VX_TIMEOUT_INFINITE))) return ret;
  if ((ret = akili_read_core_counters(device, &c1, &i1))) return ret;
  auto& perf = akili_perf();
  ++perf.launches;
  perf.cycles += c1 - c0;
  perf.instrs += i1 - i0;
  return 0;
}

inline int vx_dxa_program_desc_2d(
    vx_device_h device, uint32_t slot, uint64_t base_addr,
    uint32_t size0, uint32_t size1, uint32_t stride0_bytes,
    uint32_t tile0, uint32_t tile1, uint32_t elem_bytes) {
  uint32_t elem_size = 0;
  for (uint32_t value = elem_bytes; value > 1; value >>= 1)
    ++elem_size;
  uint32_t dcr = VX_DCR_DXA_DESC_BASE + slot * VX_DCR_DXA_DESC_STRIDE;
  uint32_t meta = (2u << DXA_DESC_META_DIM_LSB)
                | (elem_size << DXA_DESC_META_ELEMSZ_LSB);
  const struct { uint32_t off; uint32_t value; } writes[] = {
    {VX_DCR_DXA_DESC_BASE_LO_OFF, (uint32_t)base_addr},
    {VX_DCR_DXA_DESC_BASE_HI_OFF, (uint32_t)(base_addr >> 32)},
    {VX_DCR_DXA_DESC_SIZE0_OFF, size0},
    {VX_DCR_DXA_DESC_SIZE1_OFF, size1},
    {VX_DCR_DXA_DESC_STRIDE0_OFF, stride0_bytes},
    {VX_DCR_DXA_DESC_META_OFF, meta},
    {VX_DCR_DXA_DESC_ESTRIDE0_OFF, 1},
    {VX_DCR_DXA_DESC_ESTRIDE1_OFF, 1},
    {VX_DCR_DXA_DESC_TILESIZE01_OFF, (tile1 << 16) | (tile0 & 0xffffu)},
    {VX_DCR_DXA_DESC_CFILL_OFF, 0},
  };
  for (const auto& write : writes) {
    int ret = vx_dcr_write(device, dcr + write.off, write.value);
    if (ret)
      return ret;
  }
  return 0;
}

inline int akili_dxa_set_kmajor(vx_device_h device, uint32_t slot,
                                uint32_t elem_bytes) {
  uint32_t elem_size = 0;
  for (uint32_t value = elem_bytes; value > 1; value >>= 1)
    ++elem_size;
  uint32_t meta = (2u << DXA_DESC_META_DIM_LSB)
                | (elem_size << DXA_DESC_META_ELEMSZ_LSB)
                | (static_cast<uint32_t>(vortex::dxa::Layout::KMajor)
                   << DXA_DESC_META_LAYOUT_LSB);
  uint32_t dcr = VX_DCR_DXA_DESC_BASE + slot * VX_DCR_DXA_DESC_STRIDE;
  return vx_dcr_write(device, dcr + VX_DCR_DXA_DESC_META_OFF, meta);
}

// Under AKILI_META_DXA, program the metadata descriptor paired with a sparse
// GEMM's A descriptor: [M/xtileM rows][K/tileK_sp tiles x NT words] of uint32,
// fetched KB tiles x CTA-warps rows per stage. No-op otherwise.
template <typename WgCfg>
inline int akili_wg_program_meta(vx_device_h device, uint32_t desc_a,
                                 uint64_t meta_addr, uint32_t M, uint32_t K,
                                 uint32_t cta_m) {
#ifdef AKILI_META_DXA
  uint32_t row_words = (K / WgCfg::tileK_sp) * VX_CFG_NUM_THREADS;
  return vx_dxa_program_desc_2d(device, akili::wg_meta_desc(desc_a), meta_addr,
      row_words, M / WgCfg::xtileM, row_words * sizeof(uint32_t),
      akili::kWgKB * VX_CFG_NUM_THREADS, cta_m / WgCfg::xtileM, sizeof(uint32_t));
#else
  (void)device; (void)desc_a; (void)meta_addr; (void)M; (void)K; (void)cta_m;
  return 0;
#endif
}

#endif
