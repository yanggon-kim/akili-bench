// Host-side harness shared by the e2e applications: device setup, buffers,
// DXA descriptors for the padded TCU/SIMT staging layouts, per-phase MPM
// accounting and the machine-readable report consumed by the sweep script.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <VX_types.h>
#include <vortex.h>
#include <dxa.h>
#include "e2e_fmt.h"

namespace e2e {

#define E2E_CHECK(expr) do {                                              \
    vx_result_t _r = (vx_result_t)(expr);                                 \
    if (_r != VX_SUCCESS) {                                               \
      std::fprintf(stderr, "FAIL %s:%d: '%s' returned %d\n",              \
                   __FILE__, __LINE__, #expr, (int)_r);                   \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

// Counters sampled around every launch (core class; zero without PERF_ENABLE).
static const uint32_t kPhaseCsrs[] = {
  VX_CSR_MCYCLE, VX_CSR_MINSTRET,
  VX_CSR_MPM_INSTR_ALU, VX_CSR_MPM_INSTR_FPU, VX_CSR_MPM_INSTR_LSU,
  VX_CSR_MPM_INSTR_SFU, VX_CSR_MPM_INSTR_TCU,
  VX_CSR_MPM_STALL_SCRB, VX_CSR_MPM_STALL_LSU, VX_CSR_MPM_STALL_FPU,
  VX_CSR_MPM_STALL_TCU, VX_CSR_MPM_SCHED_IDLE, VX_CSR_MPM_LOADS, VX_CSR_MPM_STORES,
};
static const char* kPhaseNames[] = {
  "cycles", "instrs", "i_alu", "i_fpu", "i_lsu", "i_sfu", "i_tcu",
  "st_scrb", "st_lsu", "st_fpu", "st_tcu", "sched_idle", "loads", "stores",
};
constexpr uint32_t kNumPhaseCsrs = sizeof(kPhaseCsrs) / sizeof(kPhaseCsrs[0]);

struct Phase {
  uint64_t v[kNumPhaseCsrs] = {};
  uint32_t launches = 0;
};

class Runtime {
public:
  vx_device_h dev = nullptr;
  vx_queue_h q = nullptr;
  vx_module_h mod = nullptr;
  std::vector<vx_buffer_h> bufs;
  std::vector<std::string> order;
  std::map<std::string, Phase> phases;

  explicit Runtime(const char* kernel_file) {
    E2E_CHECK(vx_device_open(0, &dev));
    vx_queue_info_t qi = {};
    qi.struct_size = sizeof(qi);
    qi.priority = VX_QUEUE_PRIORITY_NORMAL;
    E2E_CHECK(vx_queue_create(dev, &qi, &q));
    E2E_CHECK(vx_module_load_file(dev, kernel_file, &mod));
    uint64_t nt = 0, nw = 0, nc = 0, lmem = 0, isa = 0;
    E2E_CHECK(vx_device_query(dev, VX_CAPS_NUM_THREADS, &nt));
    E2E_CHECK(vx_device_query(dev, VX_CAPS_NUM_WARPS, &nw));
    E2E_CHECK(vx_device_query(dev, VX_CAPS_NUM_CORES, &nc));
    E2E_CHECK(vx_device_query(dev, VX_CAPS_LOCAL_MEM_SIZE, &lmem));
    E2E_CHECK(vx_device_query(dev, VX_CAPS_ISA_FLAGS, &isa));
    std::printf("E2E_DEVICE cores=%lu warps=%lu threads=%lu lmem=%lu fmt=%s\n",
                (unsigned long)nc, (unsigned long)nw, (unsigned long)nt,
                (unsigned long)lmem, F::name);
    if (nt != VX_CFG_NUM_THREADS) {
      std::fprintf(stderr, "device NT=%lu != build NT=%d\n", (unsigned long)nt, VX_CFG_NUM_THREADS);
      std::exit(1);
    }
    if ((isa & VX_ISA_EXT_DXA) == 0) {
      std::fprintf(stderr, "DXA extension disabled\n");
      std::exit(1);
    }
#if E2E_TCU
    if ((isa & VX_ISA_EXT_TCU) == 0) {
      std::fprintf(stderr, "TCU extension disabled\n");
      std::exit(1);
    }
#endif
  }

  ~Runtime() {
    for (auto b : bufs) vx_buffer_release(b);
    if (mod) vx_module_release(mod);
    if (q) vx_queue_release(q);
    if (dev) vx_device_release(dev);
  }

  vx_kernel_h kernel(const char* name) {
    vx_kernel_h k = nullptr;
    E2E_CHECK(vx_module_get_kernel(mod, name, &k));
    return k;
  }

  uint64_t alloc(uint64_t bytes, vx_buffer_h* out = nullptr) {
    vx_buffer_h b = nullptr;
    // Slack past the end: GEMM mainloops may prefetch one K-tile beyond it.
    bytes = ((bytes + 63) & ~uint64_t(63)) + 16384;
    E2E_CHECK(vx_buffer_create(dev, bytes, VX_MEM_READ_WRITE, &b));
    bufs.push_back(b);
    if (out) *out = b;
    uint64_t addr = 0;
    E2E_CHECK(vx_buffer_address(b, &addr));
    return addr;
  }

  vx_buffer_h find(uint64_t addr) {
    for (auto b : bufs) {
      uint64_t a = 0;
      vx_buffer_address(b, &a);
      if (a == addr) return b;
    }
    std::fprintf(stderr, "unknown buffer 0x%lx\n", (unsigned long)addr);
    std::exit(1);
  }

  uint64_t upload(const void* data, uint64_t bytes) {
    vx_buffer_h b;
    uint64_t addr = alloc(bytes, &b);
    vx_event_h ev = nullptr;
    E2E_CHECK(vx_enqueue_write(q, b, 0, data, bytes, 0, nullptr, &ev));
    E2E_CHECK(vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(ev);
    return addr;
  }

  template <typename T>
  uint64_t upload(const std::vector<T>& v) { return upload(v.data(), v.size() * sizeof(T)); }

  void zero(uint64_t addr, uint64_t bytes) {
    std::vector<uint8_t> z(bytes, 0);
    write(addr, z.data(), bytes);
  }

  void write(uint64_t addr, const void* data, uint64_t bytes) {
    vx_event_h ev = nullptr;
    E2E_CHECK(vx_enqueue_write(q, find(addr), 0, data, bytes, 0, nullptr, &ev));
    E2E_CHECK(vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(ev);
  }

  void read(void* dst, uint64_t addr, uint64_t bytes) {
    vx_event_h ev = nullptr;
    E2E_CHECK(vx_enqueue_read(q, dst, find(addr), 0, bytes, 0, nullptr, &ev));
    E2E_CHECK(vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(ev);
  }

  void sample(uint64_t* out) {
    for (uint32_t i = 0; i < kNumPhaseCsrs; ++i) {
      uint32_t cls = (i < 2) ? VX_DCR_MPM_CLASS_BASE : VX_DCR_MPM_CLASS_CORE;
      out[i] = 0;
      E2E_CHECK(vx_device_mpm_query(dev, cls, kPhaseCsrs[i], 0xffffffff, &out[i]));
    }
  }

  // Launches one kernel, waits for it and charges the counter deltas to `phase`.
  void launch(const char* phase, vx_kernel_h k, const void* args, size_t args_size,
              uint32_t ndim, const uint32_t* grid, const uint32_t* block, uint32_t lmem) {
    uint64_t before[kNumPhaseCsrs], after[kNumPhaseCsrs];
    sample(before);
    vx_launch_info_t li = {};
    li.struct_size = sizeof(li);
    li.kernel = k;
    li.args_host = args;
    li.args_size = args_size;
    li.ndim = ndim;
    for (uint32_t d = 0; d < ndim; ++d) {
      li.grid_dim[d] = grid[d];
      li.block_dim[d] = block[d];
    }
    li.lmem_size = lmem;
    E2E_CHECK(vx_check_occupancy(dev, block[0] * (ndim > 1 ? block[1] : 1), lmem));
    vx_event_h ev = nullptr;
    E2E_CHECK(vx_enqueue_launch(q, &li, 0, nullptr, &ev));
    E2E_CHECK(vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(ev);
    sample(after);
    if (!phases.count(phase)) order.push_back(phase);
    auto& p = phases[phase];
    for (uint32_t i = 0; i < kNumPhaseCsrs; ++i) p.v[i] += after[i] - before[i];
    ++p.launches;
    std::printf("E2E_LAUNCH name=%s cycles=%lu\n", phase, (unsigned long)(after[0] - before[0]));
    std::fflush(stdout);
  }

  template <typename A>
  void launch1(const char* phase, vx_kernel_h k, const A& a, uint32_t grid, uint32_t block, uint32_t lmem = 0) {
    launch(phase, k, &a, sizeof(A), 1, &grid, &block, lmem);
  }
  template <typename A>
  void launch2(const char* phase, vx_kernel_h k, const A& a, uint32_t gx, uint32_t gy,
               uint32_t block, uint32_t lmem = 0) {
    uint32_t g[2] = {gx, gy}, b[2] = {block, 1};
    launch(phase, k, &a, sizeof(A), 2, g, b, lmem);
  }

  // ---- DXA descriptors for CTA-blocked operands ----
  // Data viewed as [rows*num_kt][16 words]; each 16-word row lands in a
  // `pitch`-word smem row, the overhang filled with the OOB value.
  void desc_blocked(uint32_t slot, uint64_t base, uint32_t rows, uint32_t num_kt,
                    uint32_t pitch_words, uint32_t tile_rows) {
    E2E_CHECK(vortex::dxa::program_2d(dev, slot, base, KT_WORDS, rows * num_kt, KT_WORDS * 4,
                                      pitch_words, tile_rows, 4));
  }
  // Scales viewed as [(rows/16)*num_kt][32 words].
  void desc_blocked_scales(uint32_t slot, uint64_t base, uint32_t rows, uint32_t num_kt,
                           uint32_t tile_rows) {
    E2E_CHECK(vortex::dxa::program_2d(dev, slot, base, 32, rows / 16 * num_kt, 128,
                                      32, tile_rows / 16, 4));
  }

  void report(double err_ref, double err_fp32, bool pass) {
    Phase tot;
    for (auto& n : order) {
      auto& p = phases[n];
      std::printf("E2E_PHASE name=%s launches=%u", n.c_str(), p.launches);
      for (uint32_t i = 0; i < kNumPhaseCsrs; ++i) {
        std::printf(" %s=%lu", kPhaseNames[i], (unsigned long)p.v[i]);
        tot.v[i] += p.v[i];
      }
      std::printf("\n");
      tot.launches += p.launches;
    }
    std::printf("E2E_TOTAL launches=%u", tot.launches);
    for (uint32_t i = 0; i < kNumPhaseCsrs; ++i)
      std::printf(" %s=%lu", kPhaseNames[i], (unsigned long)tot.v[i]);
    std::printf("\n");
    // Whole-run memory-system counters (D-cache, shared memory, DRAM).
    {
      struct { uint32_t cls, csr; const char* name; } mc[] = {
        {VX_DCR_MPM_CLASS_DCACHE, VX_CSR_MPM_DCACHE_READS, "dc_reads"},
        {VX_DCR_MPM_CLASS_DCACHE, VX_CSR_MPM_DCACHE_WRITES, "dc_writes"},
        {VX_DCR_MPM_CLASS_DCACHE, VX_CSR_MPM_DCACHE_MISS_R, "dc_miss_r"},
        {VX_DCR_MPM_CLASS_DCACHE, VX_CSR_MPM_DCACHE_MISS_W, "dc_miss_w"},
        {VX_DCR_MPM_CLASS_DCACHE, VX_CSR_MPM_DCACHE_BANK_ST, "dc_bank_st"},
        {VX_DCR_MPM_CLASS_DCACHE, VX_CSR_MPM_DCACHE_MSHR_ST, "dc_mshr_st"},
        {VX_DCR_MPM_CLASS_MEM, VX_CSR_MPM_LMEM_READS, "lmem_reads"},
        {VX_DCR_MPM_CLASS_MEM, VX_CSR_MPM_LMEM_WRITES, "lmem_writes"},
        {VX_DCR_MPM_CLASS_MEM, VX_CSR_MPM_LMEM_BANK_ST, "lmem_bank_st"},
        {VX_DCR_MPM_CLASS_MEM, VX_CSR_MPM_MEM_READS, "dram_reads"},
        {VX_DCR_MPM_CLASS_MEM, VX_CSR_MPM_MEM_WRITES, "dram_writes"},
        {VX_DCR_MPM_CLASS_MEM, VX_CSR_MPM_MEM_LT, "dram_lat_sum"},
      };
      std::printf("E2E_MEM");
      for (auto& m : mc) {
        uint64_t v = 0;
        vx_device_mpm_query(dev, m.cls, m.csr, 0, &v);
        std::printf(" %s=%lu", m.name, (unsigned long)v);
      }
      std::printf("\n");
    }
    std::printf("E2E_ACC err_vs_emulated=%.6e err_vs_fp32=%.6e status=%s\n",
                err_ref, err_fp32, pass ? "PASS" : "FAIL");
    vx_device_dump_perf(dev, stdout);
    std::printf(pass ? "PASSED!\n" : "FAILED!\n");
  }
};

// ---- host-side quantization of operands ----
// Quantizes a row-major [rows x K] fp32 matrix into the CTA-blocked operand
// layout (block = `rb` rows) plus blocked scale tiles. K must be a multiple
// of KT and rows of rb (pad first).
struct QTensor {
  uint32_t rows = 0, k = 0, rb = 0;
  float tscale = 1.0f;
  std::vector<uint32_t> words;   // blocked, rows * k / epw
  std::vector<uint8_t> scales;   // blocked, (rows/16) * (k/KT) * 128 (MX/NV only)
  std::vector<float> deq;        // row-major dequantized values (host reference)
  uint32_t num_kt() const { return k / KT; }
};

inline QTensor quantize_host(const float* x, uint32_t rows, uint32_t k, float tscale, uint32_t rb) {
  QTensor t;
  t.rows = rows; t.k = k; t.rb = rb; t.tscale = tscale;
  const uint32_t nkt = k / KT, rw = k / F::epw;
  t.words.assign((size_t)rows * rw, 0);
  if (IS_MX) t.scales.assign((size_t)((rows + rb - 1) / rb) * rb / 16 * nkt * SC_TILE_BYTES, 0);
  t.deq.assign((size_t)rows * k, 0.0f);
  std::vector<uint32_t> row_words(rw);
  std::vector<uint8_t> sc(k / BLK + 1);
  const uint32_t mask = F::bits == 32 ? 0xffffffffu : ((1u << F::bits) - 1);
  for (uint32_t r = 0; r < rows; ++r) {
    quantize_run(x + (size_t)r * k, k, tscale, row_words.data(), sc.data());
    for (uint32_t w = 0; w < rw; ++w) t.words[blk_word(r, w, rb, nkt)] = row_words[w];
    for (uint32_t kk = 0; kk < k; ++kk) {
      uint32_t code = (row_words[kk / F::epw] >> ((kk % F::epw) * F::bits)) & mask;
      uint32_t sf = IS_MX ? sc[kk / BLK] : 0;
      t.deq[(size_t)r * k + kk] = decode(code, sf, tscale);
      if (IS_MX && (kk % BLK) == 0) t.scales[blk_sc(r, kk, rb, nkt)] = (uint8_t)sf;
    }
  }
  return t;
}

// Quantize one row of n values (n a multiple of the scale block) and return
// the dequantized values (what the device's quant_store_run produces).
inline void quant_dequant_row(const float* x, uint32_t n, float tscale, float* out) {
  std::vector<uint32_t> w(n / F::epw + 1);
  std::vector<uint8_t> sc(n / BLK + 1);
  quantize_run(x, n, tscale, w.data(), sc.data());
  const uint32_t mask = F::bits == 32 ? 0xffffffffu : ((1u << F::bits) - 1);
  for (uint32_t i = 0; i < n; ++i)
    out[i] = decode((w[i / F::epw] >> ((i % F::epw) * F::bits)) & mask, IS_MX ? sc[i / BLK] : 0, tscale);
}

// Dequantize a device-produced blocked operand back to row-major fp32.
inline std::vector<float> dequantize_host(const std::vector<uint32_t>& words,
                                          const std::vector<uint8_t>& scales,
                                          uint32_t rows, uint32_t k, float tscale, uint32_t rb) {
  const uint32_t nkt = k / KT;
  const uint32_t mask = F::bits == 32 ? 0xffffffffu : ((1u << F::bits) - 1);
  std::vector<float> out((size_t)rows * k);
  for (uint32_t r = 0; r < rows; ++r)
    for (uint32_t kk = 0; kk < k; ++kk) {
      uint32_t w = words[blk_word(r, kk / F::epw, rb, nkt)];
      uint32_t code = (w >> ((kk % F::epw) * F::bits)) & mask;
      uint32_t sf = IS_MX ? scales[blk_sc(r, kk, rb, nkt)] : 0;
      out[(size_t)r * k + kk] = decode(code, sf, tscale);
    }
  return out;
}

inline double rel_rmse(const float* a, const float* ref, size_t n) {
  double num = 0, den = 0;
  for (size_t i = 0; i < n; ++i) {
    double d = (double)a[i] - (double)ref[i];
    num += d * d;
    den += (double)ref[i] * ref[i];
  }
  return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
}

inline uint32_t round_up(uint32_t x, uint32_t m) { return (x + m - 1) / m * m; }

// Deterministic uniform [-1, 1).
struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 1) {}
  float next() {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    return (float)((s >> 40) & 0xffffff) / 8388608.0f - 1.0f;
  }
};

} // namespace e2e
