// akili_NeRF_tcu_sp — Full NeRF forward pass with the MLP GEMMs running on
// the SPARSE (2:4) tensor core.
//
//   Stage 1 (SIMT):   ray-AABB slab + stratified sampling + PE
//   Stage 2 (TCU_SP): 4 MLP layer GEMMs (W_compressed x X -> Y_fp32)
//   Stage 2 (SIMT):   ReLU + bias + fp32->fp16 cast (4x)
//   Stage 3 (SIMT):   per-ray alpha compositing
//
// Host-side per layer: fp16 convert -> prune_2to4_matrix -> compress_2to4_matrix
// -> pack_metadata -> upload compressed weights + metadata.
// CPU reference uses the PRUNED fp32 weights so device vs CPU match.

#include <iostream>
#include <unistd.h>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <vortex.h>
#include <cmath>
#include <algorithm>
#include <stdio.h>
#include "common.h"

#ifndef NUM_THREADS
#define NUM_THREADS 8
#endif
#ifndef NUM_TCU_LANES
#define NUM_TCU_LANES NUM_THREADS
#endif

#include <tensor_cfg.h>
#include <tensor.h>   // prune_2to4_matrix, compress_2to4_matrix
namespace vt = vortex::tensor;
using cfg = vt::wmma_config_t<NUM_TCU_LANES, vt::fp16, vt::fp32>;
static constexpr uint32_t TM = cfg::tileM;
static constexpr uint32_t TN = cfg::tileN;
static constexpr uint32_t TK = cfg::tileK;

// ---------------------------------------------------------------------------
// pack_metadata — lifted verbatim from akili_attn_tcu_sp/main.cpp.
// Parameterized by (M, K): works for each layer's weight matrix.
// ---------------------------------------------------------------------------
static void pack_metadata(std::vector<uint32_t>& h_meta,
                          const std::vector<uint8_t>& masks,
                          uint32_t M, uint32_t K) {
  constexpr uint32_t I_RATIO = cfg::rtl_i_ratio;
  constexpr uint32_t TC_K = cfg::tcK;
  constexpr uint32_t TC_M = cfg::tcM;
  constexpr uint32_t meta_row_w = TC_K * 2 * I_RATIO;
  constexpr uint32_t mcols = cfg::meta_cols;
  constexpr uint32_t half_k_steps = cfg::k_steps / 2;
  constexpr uint32_t PD = cfg::m_steps * (cfg::k_steps / 2);
  constexpr uint32_t banks_per_store = (NUM_THREADS < PD) ? NUM_THREADS : PD;
  constexpr uint32_t stores_per_col = (PD + NUM_THREADS - 1) / NUM_THREADS;
  constexpr uint32_t cols_per_load = (NUM_THREADS >= PD) ? (NUM_THREADS / PD) : 1;
  constexpr uint32_t num_meta_loads = (PD * mcols + NUM_THREADS - 1) / NUM_THREADS;
  constexpr uint32_t per_k_tile_words = num_meta_loads * NUM_THREADS;

  uint32_t tileK_elem = cfg::tileK;
  uint32_t num_groups_per_row = K / 4;
  uint32_t elts_per_sparse_step = tileK_elem / half_k_steps;

  uint32_t num_tile_rows = M / cfg::tileM;
  uint32_t num_k_tiles = K / cfg::tileK;

  h_meta.assign(num_tile_rows * num_k_tiles * per_k_tile_words, 0);

  for (uint32_t tr = 0; tr < num_tile_rows; ++tr) {
    for (uint32_t kt = 0; kt < num_k_tiles; ++kt) {
      uint32_t section_base = (tr * num_k_tiles + kt) * per_k_tile_words;
      for (uint32_t sm = 0; sm < cfg::m_steps; ++sm) {
        for (uint32_t sk = 0; sk < half_k_steps; ++sk) {
          uint32_t sram_row = sm * half_k_steps + sk;
          for (uint32_t i = 0; i < TC_M; ++i) {
            uint32_t physical_row = tr * cfg::tileM + sm * TC_M + i;
            uint32_t k_elem_start = kt * tileK_elem + sk * elts_per_sparse_step;
            for (uint32_t e = 0; e < elts_per_sparse_step; ++e) {
              uint32_t global_elt = k_elem_start + e;
              uint32_t global_group = global_elt / 4;
              uint32_t pos_in_group = global_elt % 4;
              uint8_t mask = masks[physical_row * num_groups_per_row + global_group];
              if (mask & (1u << pos_in_group)) {
                uint32_t k_reg = e / (2 * I_RATIO);
                uint32_t pos_in_k = e % (2 * I_RATIO);
                uint32_t meta_bit;
                if (pos_in_k < I_RATIO) {
                  meta_bit = k_reg * I_RATIO + pos_in_k;
                } else {
                  meta_bit = (TC_K + k_reg) * I_RATIO + (pos_in_k - I_RATIO);
                }
                uint32_t block_bit = i * meta_row_w + meta_bit;
                uint32_t word_idx = block_bit / 32;
                uint32_t bit_idx = block_bit % 32;
                uint32_t store_in_col = sram_row / banks_per_store;
                uint32_t thread_in_store = sram_row % banks_per_store;
                uint32_t flat_store = word_idx * stores_per_col + store_in_col;
                uint32_t load_idx = flat_store / cols_per_load;
                uint32_t store_in_load = flat_store % cols_per_load;
                uint32_t meta_idx = load_idx * NUM_THREADS + store_in_load * banks_per_store + thread_in_store;
                h_meta[section_base + meta_idx] |= (1u << bit_idx);
              }
            }
          }
        }
      }
    }
  }
}

#define RT_CHECK(_expr)                                         \
   do {                                                         \
     int _ret = _expr;                                          \
     if (0 == _ret) break;                                      \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
     cleanup();                                                 \
     exit(-1);                                                  \
   } while (false)

// ---------------------------------------------------------------------------
// Host-side fp32 <-> fp16
// ---------------------------------------------------------------------------
static inline uint16_t f2h_host(float x) {
  uint32_t bits;
  std::memcpy(&bits, &x, sizeof(bits));
  uint16_t sign = (bits >> 16) & 0x8000;
  uint16_t mant = (bits >> 13) & 0x03FF;
  int32_t  ex   = (int32_t)((bits >> 23) & 0xFF) - 127 + 15;
  if (ex >= 31) return sign | 0x7C00;
  if (ex <= 0)  return sign;
  return sign | ((uint16_t)ex << 10) | mant;
}
static inline float h2f_host(uint16_t x) {
  uint32_t sign = (uint32_t)(x & 0x8000) << 16;
  uint32_t ex   = (x >> 10) & 0x1F;
  uint32_t mant = (x & 0x3FF);
  uint32_t bits;
  if (ex == 0)       bits = sign;
  else if (ex == 31) bits = sign | 0x7F800000 | (mant << 13);
  else               bits = sign | ((ex + 127 - 15) << 23) | (mant << 13);
  float f; std::memcpy(&f, &bits, sizeof(f)); return f;
}

static bool compare(float a, float b, float atol, float rtol,
                    int index, int& errors, const char* tag) {
  auto diff = std::fabs(a - b);
  auto limit = atol + rtol * std::fabs(b);
  if (diff > limit) {
    if (errors < 20)
      printf("*** %s error: [%d] expected=%f, actual=%f (diff=%g)\n",
             tag, index, a, b, (float)diff);
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// CPU reference — uses the PRUNED fp32 weights (matches what the device sees
// after 2:4 compression).
// ---------------------------------------------------------------------------
static void ray_setup_cpu(float* pe, float* deltas,
                          const float* rays_o, const float* rays_d,
                          const float* aabb, uint32_t n_rays,
                          uint32_t n_samples, float min_near) {
  uint32_t n_pts = n_rays * n_samples;
  for (uint32_t ray = 0; ray < n_rays; ++ray) {
    float ox = rays_o[ray*3+0], oy = rays_o[ray*3+1], oz = rays_o[ray*3+2];
    float dx = rays_d[ray*3+0], dy = rays_d[ray*3+1], dz = rays_d[ray*3+2];
    float inv_dx = 1.0f/dx, inv_dy = 1.0f/dy, inv_dz = 1.0f/dz;
    float tx1 = (aabb[0]-ox)*inv_dx, tx2 = (aabb[3]-ox)*inv_dx;
    float ty1 = (aabb[1]-oy)*inv_dy, ty2 = (aabb[4]-oy)*inv_dy;
    float tz1 = (aabb[2]-oz)*inv_dz, tz2 = (aabb[5]-oz)*inv_dz;
    float tmin_x = std::min(tx1, tx2), tmax_x = std::max(tx1, tx2);
    float tmin_y = std::min(ty1, ty2), tmax_y = std::max(ty1, ty2);
    float tmin_z = std::min(tz1, tz2), tmax_z = std::max(tz1, tz2);
    float tmin = std::max(std::max(tmin_x, tmin_y), tmin_z);
    float tmax = std::min(std::min(tmax_x, tmax_y), tmax_z);
    if (tmin < min_near) tmin = min_near;
    float step = 0.0f;
    if (tmax > tmin) step = (tmax - tmin) / (float)n_samples;
    else             tmax = tmin;
    for (uint32_t s = 0; s < n_samples; ++s) {
      uint32_t pt = ray * n_samples + s;
      float t  = tmin + ((float)s + 0.5f) * step;
      float px = ox + t*dx, py = oy + t*dy, pz = oz + t*dz;
      deltas[pt] = step;
      pe[0*n_pts + pt] = px;
      pe[1*n_pts + pt] = py;
      pe[2*n_pts + pt] = pz;
      float freq = 1.0f;
      for (int L = 0; L < PE_L; ++L) {
        uint32_t base = 3 + L*6;
        pe[(base+0)*n_pts + pt] = sinf(freq * px);
        pe[(base+1)*n_pts + pt] = cosf(freq * px);
        pe[(base+2)*n_pts + pt] = sinf(freq * py);
        pe[(base+3)*n_pts + pt] = cosf(freq * py);
        pe[(base+4)*n_pts + pt] = sinf(freq * pz);
        pe[(base+5)*n_pts + pt] = cosf(freq * pz);
        freq *= 2.0f;
      }
      for (uint32_t f = MLP_IN_DIM; f < MLP_IN_DIM_PAD; ++f)
        pe[f*n_pts + pt] = 0.0f;
    }
  }
}

static void layer_gemm_cpu(float* C, const float* W, const float* X,
                           uint32_t N_out, uint32_t K_in, uint32_t n_pts) {
  for (uint32_t m = 0; m < N_out; ++m) {
    for (uint32_t n = 0; n < n_pts; ++n) {
      double sum = 0.0;
      for (uint32_t k = 0; k < K_in; ++k)
        sum += (double)W[m * K_in + k] * (double)X[k * n_pts + n];
      C[m * n_pts + n] = (float)sum;
    }
  }
}

// ---------------------------------------------------------------------------
const char* kernel_file = "kernel.vxbin";
uint32_t n_rays_req    = 32;
uint32_t n_samples_req = 8;

vx_device_h device = nullptr;
vx_buffer_h rays_o_buf = nullptr;
vx_buffer_h rays_d_buf = nullptr;
vx_buffer_h aabb_buf   = nullptr;
vx_buffer_h pe_buf     = nullptr;
vx_buffer_h delta_buf  = nullptr;
// Sparse variant: compressed weights + metadata per layer.
vx_buffer_h Wc0_buf = nullptr, Wc1_buf = nullptr, Wc2_buf = nullptr, Wc3_buf = nullptr;
vx_buffer_h meta0_buf = nullptr, meta1_buf = nullptr, meta2_buf = nullptr, meta3_buf = nullptr;
vx_buffer_h B0_buf = nullptr, B1_buf = nullptr, B2_buf = nullptr, B3_buf = nullptr;
vx_buffer_h act_a_buf = nullptr, act_b_buf = nullptr;
vx_buffer_h scratch_fp32_buf = nullptr;
vx_buffer_h out_head_fp32_buf = nullptr;
vx_buffer_h sigma_buf = nullptr, rgb_buf = nullptr, image_buf = nullptr;
vx_buffer_h cycles_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

uint64_t Wc0_addr=0, Wc1_addr=0, Wc2_addr=0, Wc3_addr=0;
uint64_t M0_addr=0,  M1_addr=0,  M2_addr=0,  M3_addr=0;
uint64_t B0_addr=0,  B1_addr=0,  B2_addr=0,  B3_addr=0;
uint64_t pe_addr=0,  act_a_addr=0, act_b_addr=0;
uint64_t scratch_fp32_addr=0, out_head_fp32_addr=0;

static void show_usage() {
  std::cout << "akili_NeRF_tcu_sp — Sparse (2:4) TCU full-NeRF forward" << std::endl;
  std::cout << "Usage: [-r rays] [-s samples_per_ray] [-k kernel_file]" << std::endl;
  std::cout << "Defaults: -r 32 -s 8" << std::endl;
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "r:s:k:h")) != -1) {
    switch (c) {
      case 'r': n_rays_req    = atoi(optarg); break;
      case 's': n_samples_req = atoi(optarg); break;
      case 'k': kernel_file   = optarg;       break;
      case 'h': show_usage(); exit(0);
      default:  show_usage(); exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    if (rays_o_buf) vx_mem_free(rays_o_buf);
    if (rays_d_buf) vx_mem_free(rays_d_buf);
    if (aabb_buf)   vx_mem_free(aabb_buf);
    if (pe_buf)     vx_mem_free(pe_buf);
    if (delta_buf)  vx_mem_free(delta_buf);
    if (Wc0_buf) vx_mem_free(Wc0_buf);
    if (Wc1_buf) vx_mem_free(Wc1_buf);
    if (Wc2_buf) vx_mem_free(Wc2_buf);
    if (Wc3_buf) vx_mem_free(Wc3_buf);
    if (meta0_buf) vx_mem_free(meta0_buf);
    if (meta1_buf) vx_mem_free(meta1_buf);
    if (meta2_buf) vx_mem_free(meta2_buf);
    if (meta3_buf) vx_mem_free(meta3_buf);
    if (B0_buf) vx_mem_free(B0_buf);
    if (B1_buf) vx_mem_free(B1_buf);
    if (B2_buf) vx_mem_free(B2_buf);
    if (B3_buf) vx_mem_free(B3_buf);
    if (act_a_buf) vx_mem_free(act_a_buf);
    if (act_b_buf) vx_mem_free(act_b_buf);
    if (scratch_fp32_buf)  vx_mem_free(scratch_fp32_buf);
    if (out_head_fp32_buf) vx_mem_free(out_head_fp32_buf);
    if (sigma_buf) vx_mem_free(sigma_buf);
    if (rgb_buf)   vx_mem_free(rgb_buf);
    if (image_buf) vx_mem_free(image_buf);
    if (cycles_buffer) vx_mem_free(cycles_buffer);
    if (krnl_buffer) vx_mem_free(krnl_buffer);
    if (args_buffer) vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

static uint64_t read_back_cycles_tag(const char* tag, uint32_t num_blocks, bool print) {
  std::vector<uint32_t> h_cycles(num_blocks, 0);
  vx_copy_from_dev(h_cycles.data(), cycles_buffer, 0, num_blocks * sizeof(uint32_t));
  uint32_t max_cyc = 0;
  for (auto c : h_cycles) if (c > max_cyc) max_cyc = c;
  if (print)
    printf("KCYC[%s,nt=%u]: %u\n", tag, (unsigned)NUM_THREADS, max_cyc);
  return (uint64_t)max_cyc;
}

// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
  parse_args(argc, argv);

  std::cout << "open device connection" << std::endl;
  RT_CHECK(vx_dev_open(&device));

  uint64_t NT_caps = 0;
  vx_dev_caps(device, VX_CAPS_NUM_THREADS, &NT_caps);
  if ((uint32_t)NT_caps != NUM_THREADS) {
    printf("Error: device NUM_THREADS=%u but built with NUM_THREADS=%u\n",
           (unsigned)NT_caps, (unsigned)NUM_THREADS);
    cleanup();
    return -1;
  }

  uint32_t n_rays    = n_rays_req;
  uint32_t n_samples = n_samples_req;
  uint32_t n_points  = n_rays * n_samples;
  if (n_points % TN != 0) {
    printf("Error: n_rays*n_samples=%u must be a multiple of TN=%u\n",
           n_points, TN);
    cleanup();
    return -1;
  }
  std::cout << "akili_NeRF_tcu_sp — n_rays=" << n_rays
            << " n_samples=" << n_samples
            << " n_points=" << n_points
            << " MLP=" << MLP_DEPTH << "x" << MLP_W
            << " PE_L=" << PE_L
            << " TM=" << TM << " TN=" << TN << " TK=" << TK << std::endl;

  // ---- Ray / camera init ----
  std::vector<float> h_rays_o(n_rays * 3);
  std::vector<float> h_rays_d(n_rays * 3);
  int gw = 1;
  while ((uint32_t)(gw * gw) < n_rays) ++gw;
  for (uint32_t i = 0; i < n_rays; ++i) {
    int px = (int)(i % (uint32_t)gw);
    int py = (int)(i / (uint32_t)gw);
    float u = ((float)px + 0.5f) / (float)gw - 0.5f;
    float v = ((float)py + 0.5f) / (float)gw - 0.5f;
    h_rays_o[i*3+0] = 0.0f;
    h_rays_o[i*3+1] = 0.0f;
    h_rays_o[i*3+2] = -2.0f;
    float dx = u, dy = v, dz = 1.0f;
    float n = 1.0f / std::sqrt(dx*dx + dy*dy + dz*dz);
    h_rays_d[i*3+0] = dx * n;
    h_rays_d[i*3+1] = dy * n;
    h_rays_d[i*3+2] = dz * n;
  }
  float h_aabb[6] = {-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f};
  float min_near = 0.5f;

  // ---- MLP weights / biases (deterministic) ----
  std::srand(42);
  auto urnd = []() {
    return 0.1f * ((float)std::rand() / (float)RAND_MAX * 2.0f - 1.0f);
  };

  std::vector<float> h_W0(MLP_W * MLP_IN_DIM_PAD, 0.0f);
  std::vector<float> h_W1(MLP_W * MLP_W, 0.0f);
  std::vector<float> h_W2(MLP_W * MLP_W, 0.0f);
  std::vector<float> h_W3(MLP_OUT_DIM_PAD * MLP_W, 0.0f);
  for (auto& v : h_W0) v = urnd();
  for (auto& v : h_W1) v = urnd();
  for (auto& v : h_W2) v = urnd();
  for (uint32_t r = 0; r < MLP_OUT_DIM; ++r)
    for (uint32_t k = 0; k < MLP_W; ++k)
      h_W3[r * MLP_W + k] = urnd();

  std::vector<float> h_B0(MLP_W, 0.0f);
  std::vector<float> h_B1(MLP_W, 0.0f);
  std::vector<float> h_B2(MLP_W, 0.0f);
  std::vector<float> h_B3(MLP_OUT_DIM_PAD, 0.0f);
  h_B3[0] = 0.5f;

  // ---- Prune + compress + pack metadata for each layer ----
  // After pruning, overwrite the fp32 host weight with the pruned version
  // so the CPU reference uses the same values as the device.
  auto prune_compress_layer =
      [](std::vector<float>& h_W,                 // [M x K] fp32, updated in place
         std::vector<uint16_t>& h_W_fp16_pruned,  // [M x K] fp16, OUT
         std::vector<uint16_t>& h_Wc,             // [M x K/2] fp16 compressed, OUT
         std::vector<uint32_t>& h_meta,           // packed metadata, OUT
         uint32_t M, uint32_t K) -> bool {
    h_W_fp16_pruned.assign(M * K, 0);
    for (uint32_t i = 0; i < M * K; ++i) h_W_fp16_pruned[i] = f2h_host(h_W[i]);
    if (!vt::prune_2to4_matrix<vt::fp16>(h_W_fp16_pruned.data(), M, K))
      return false;
    // Copy pruned values back to fp32 host vector for the CPU reference.
    for (uint32_t i = 0; i < M * K; ++i) h_W[i] = h2f_host(h_W_fp16_pruned[i]);

    h_Wc.assign(M * (K / 2), 0);
    std::vector<uint8_t> masks;
    if (!vt::compress_2to4_matrix<vt::fp16>(h_Wc.data(), h_W_fp16_pruned.data(),
                                            masks, M, K))
      return false;
    pack_metadata(h_meta, masks, M, K);
    return true;
  };

  std::vector<uint16_t> h_W0_fp16, h_W1_fp16, h_W2_fp16, h_W3_fp16;
  std::vector<uint16_t> h_Wc0, h_Wc1, h_Wc2, h_Wc3;
  std::vector<uint32_t> h_meta0, h_meta1, h_meta2, h_meta3;

  if (!prune_compress_layer(h_W0, h_W0_fp16, h_Wc0, h_meta0, MLP_W, MLP_IN_DIM_PAD)) {
    printf("prune+compress(W0) failed\n"); cleanup(); return -1;
  }
  if (!prune_compress_layer(h_W1, h_W1_fp16, h_Wc1, h_meta1, MLP_W, MLP_W)) {
    printf("prune+compress(W1) failed\n"); cleanup(); return -1;
  }
  if (!prune_compress_layer(h_W2, h_W2_fp16, h_Wc2, h_meta2, MLP_W, MLP_W)) {
    printf("prune+compress(W2) failed\n"); cleanup(); return -1;
  }
  if (!prune_compress_layer(h_W3, h_W3_fp16, h_Wc3, h_meta3, MLP_OUT_DIM_PAD, MLP_W)) {
    printf("prune+compress(W3) failed\n"); cleanup(); return -1;
  }

  // ---- Allocate device buffers ----
  uint32_t rays_bytes      = n_rays * 3 * sizeof(float);
  uint32_t aabb_bytes      = 6 * sizeof(float);
  uint32_t pe_bytes        = MLP_IN_DIM_PAD * n_points * sizeof(uint16_t);
  uint32_t delta_bytes     = n_points * sizeof(float);
  uint32_t act_bytes       = MLP_W * n_points * sizeof(uint16_t);
  uint32_t scratch_bytes   = MLP_W * n_points * sizeof(float);
  uint32_t out_head_bytes  = MLP_OUT_DIM_PAD * n_points * sizeof(float);
  uint32_t sigma_bytes     = n_points * sizeof(float);
  uint32_t rgb_bytes       = n_points * 3 * sizeof(float);
  uint32_t image_bytes     = n_rays * 3 * sizeof(float);

  RT_CHECK(vx_mem_alloc(device, rays_bytes, VX_MEM_READ_WRITE, &rays_o_buf));
  RT_CHECK(vx_mem_address(rays_o_buf, &kernel_arg.rays_o_addr));
  RT_CHECK(vx_mem_alloc(device, rays_bytes, VX_MEM_READ_WRITE, &rays_d_buf));
  RT_CHECK(vx_mem_address(rays_d_buf, &kernel_arg.rays_d_addr));
  RT_CHECK(vx_mem_alloc(device, aabb_bytes, VX_MEM_READ_WRITE, &aabb_buf));
  RT_CHECK(vx_mem_address(aabb_buf, &kernel_arg.aabb_addr));
  RT_CHECK(vx_mem_alloc(device, pe_bytes, VX_MEM_READ_WRITE, &pe_buf));
  RT_CHECK(vx_mem_address(pe_buf, &pe_addr));
  kernel_arg.pe_buf_addr = pe_addr;
  RT_CHECK(vx_mem_alloc(device, delta_bytes, VX_MEM_READ_WRITE, &delta_buf));
  RT_CHECK(vx_mem_address(delta_buf, &kernel_arg.deltas_addr));

  auto alloc_weight = [&](uint32_t nbytes, vx_buffer_h* buf, uint64_t* addr) {
    RT_CHECK(vx_mem_alloc(device, nbytes, VX_MEM_READ_WRITE, buf));
    RT_CHECK(vx_mem_address(*buf, addr));
  };
  alloc_weight((uint32_t)h_Wc0.size()*sizeof(uint16_t), &Wc0_buf, &Wc0_addr);
  alloc_weight((uint32_t)h_Wc1.size()*sizeof(uint16_t), &Wc1_buf, &Wc1_addr);
  alloc_weight((uint32_t)h_Wc2.size()*sizeof(uint16_t), &Wc2_buf, &Wc2_addr);
  alloc_weight((uint32_t)h_Wc3.size()*sizeof(uint16_t), &Wc3_buf, &Wc3_addr);
  alloc_weight((uint32_t)h_meta0.size()*sizeof(uint32_t), &meta0_buf, &M0_addr);
  alloc_weight((uint32_t)h_meta1.size()*sizeof(uint32_t), &meta1_buf, &M1_addr);
  alloc_weight((uint32_t)h_meta2.size()*sizeof(uint32_t), &meta2_buf, &M2_addr);
  alloc_weight((uint32_t)h_meta3.size()*sizeof(uint32_t), &meta3_buf, &M3_addr);
  alloc_weight(h_B0.size()*sizeof(float), &B0_buf, &B0_addr);
  alloc_weight(h_B1.size()*sizeof(float), &B1_buf, &B1_addr);
  alloc_weight(h_B2.size()*sizeof(float), &B2_buf, &B2_addr);
  alloc_weight(h_B3.size()*sizeof(float), &B3_buf, &B3_addr);

  RT_CHECK(vx_mem_alloc(device, act_bytes, VX_MEM_READ_WRITE, &act_a_buf));
  RT_CHECK(vx_mem_address(act_a_buf, &act_a_addr));
  RT_CHECK(vx_mem_alloc(device, act_bytes, VX_MEM_READ_WRITE, &act_b_buf));
  RT_CHECK(vx_mem_address(act_b_buf, &act_b_addr));
  RT_CHECK(vx_mem_alloc(device, scratch_bytes, VX_MEM_READ_WRITE, &scratch_fp32_buf));
  RT_CHECK(vx_mem_address(scratch_fp32_buf, &scratch_fp32_addr));
  RT_CHECK(vx_mem_alloc(device, out_head_bytes, VX_MEM_READ_WRITE, &out_head_fp32_buf));
  RT_CHECK(vx_mem_address(out_head_fp32_buf, &out_head_fp32_addr));

  RT_CHECK(vx_mem_alloc(device, sigma_bytes, VX_MEM_READ_WRITE, &sigma_buf));
  RT_CHECK(vx_mem_address(sigma_buf, &kernel_arg.sigmas_addr));
  RT_CHECK(vx_mem_alloc(device, rgb_bytes, VX_MEM_READ_WRITE, &rgb_buf));
  RT_CHECK(vx_mem_address(rgb_buf, &kernel_arg.rgbs_addr));
  RT_CHECK(vx_mem_alloc(device, image_bytes, VX_MEM_READ_WRITE, &image_buf));
  RT_CHECK(vx_mem_address(image_buf, &kernel_arg.image_addr));

  kernel_arg.n_rays    = n_rays;
  kernel_arg.n_samples = n_samples;
  kernel_arg.n_points  = n_points;
  kernel_arg.min_near  = min_near;

  // ---- Copy constants to device ----
  RT_CHECK(vx_copy_to_dev(rays_o_buf, h_rays_o.data(), 0, rays_bytes));
  RT_CHECK(vx_copy_to_dev(rays_d_buf, h_rays_d.data(), 0, rays_bytes));
  RT_CHECK(vx_copy_to_dev(aabb_buf,   h_aabb,          0, aabb_bytes));

  RT_CHECK(vx_copy_to_dev(Wc0_buf, h_Wc0.data(), 0, h_Wc0.size()*sizeof(uint16_t)));
  RT_CHECK(vx_copy_to_dev(Wc1_buf, h_Wc1.data(), 0, h_Wc1.size()*sizeof(uint16_t)));
  RT_CHECK(vx_copy_to_dev(Wc2_buf, h_Wc2.data(), 0, h_Wc2.size()*sizeof(uint16_t)));
  RT_CHECK(vx_copy_to_dev(Wc3_buf, h_Wc3.data(), 0, h_Wc3.size()*sizeof(uint16_t)));
  RT_CHECK(vx_copy_to_dev(meta0_buf, h_meta0.data(), 0, h_meta0.size()*sizeof(uint32_t)));
  RT_CHECK(vx_copy_to_dev(meta1_buf, h_meta1.data(), 0, h_meta1.size()*sizeof(uint32_t)));
  RT_CHECK(vx_copy_to_dev(meta2_buf, h_meta2.data(), 0, h_meta2.size()*sizeof(uint32_t)));
  RT_CHECK(vx_copy_to_dev(meta3_buf, h_meta3.data(), 0, h_meta3.size()*sizeof(uint32_t)));
  RT_CHECK(vx_copy_to_dev(B0_buf, h_B0.data(), 0, h_B0.size()*sizeof(float)));
  RT_CHECK(vx_copy_to_dev(B1_buf, h_B1.data(), 0, h_B1.size()*sizeof(float)));
  RT_CHECK(vx_copy_to_dev(B2_buf, h_B2.data(), 0, h_B2.size()*sizeof(float)));
  RT_CHECK(vx_copy_to_dev(B3_buf, h_B3.data(), 0, h_B3.size()*sizeof(float)));

  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
  RT_CHECK(vx_mem_alloc(device, sizeof(kernel_arg_t), VX_MEM_READ_WRITE, &args_buffer));

  // Per-stage grid/block shapes. Largest grid sizes the cycle buffer.
  uint32_t gd_setup      = (n_points + NUM_THREADS - 1) / NUM_THREADS;
  uint32_t gd_gemm_x     = n_points / TN;
  uint32_t gd_gemm_y_max = MLP_W / TM;
  uint32_t gd_gemm_blocks_max = gd_gemm_x * gd_gemm_y_max;
  uint32_t gd_act_max    = (MLP_W * n_points + NUM_THREADS - 1) / NUM_THREADS;
  uint32_t gd_out_act    = (n_points + NUM_THREADS - 1) / NUM_THREADS;
  uint32_t gd_comp       = (n_rays   + NUM_THREADS - 1) / NUM_THREADS;
  uint32_t max_blocks    = gd_setup;
  if (gd_gemm_blocks_max > max_blocks) max_blocks = gd_gemm_blocks_max;
  if (gd_act_max > max_blocks) max_blocks = gd_act_max;
  if (gd_out_act > max_blocks) max_blocks = gd_out_act;
  if (gd_comp    > max_blocks) max_blocks = gd_comp;
  RT_CHECK(vx_mem_alloc(device, max_blocks * sizeof(uint32_t),
                        VX_MEM_READ_WRITE, &cycles_buffer));
  RT_CHECK(vx_mem_address(cycles_buffer, &kernel_arg.cycles_addr));

  uint32_t block_dim[2] = {NUM_THREADS, 1};

  int errors = 0;
  const float atol = 1e-2f;
  const float rtol = 1e-2f;

  uint64_t cyc_setup = 0, cyc_mlp_gemm = 0, cyc_mlp_act = 0, cyc_comp = 0;

  // ================== Stage 1: ray setup ==================
  std::cout << "=== Stage 1: ray setup + positional encoding ===" << std::endl;
  kernel_arg.kernel_id = KID_RAY_SETUP;
  RT_CHECK(vx_copy_to_dev(args_buffer, &kernel_arg, 0, sizeof(kernel_arg_t)));
  {
    uint32_t grid_dim[2] = {gd_setup, 1};
    RT_CHECK(vx_start_g(device, krnl_buffer, args_buffer, 1, grid_dim, block_dim, 0));
  }
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
  cyc_setup = read_back_cycles_tag("SETUP", gd_setup, true);

  std::vector<uint16_t> h_pe_fp16(MLP_IN_DIM_PAD * n_points);
  std::vector<float>    h_deltas(n_points);
  RT_CHECK(vx_copy_from_dev(h_pe_fp16.data(), pe_buf,    0, pe_bytes));
  RT_CHECK(vx_copy_from_dev(h_deltas.data(),  delta_buf, 0, delta_bytes));
  {
    std::vector<float> ref_pe(MLP_IN_DIM_PAD * n_points);
    std::vector<float> ref_deltas(n_points);
    ray_setup_cpu(ref_pe.data(), ref_deltas.data(),
                  h_rays_o.data(), h_rays_d.data(), h_aabb,
                  n_rays, n_samples, min_near);
    int stage_err = 0;
    for (uint32_t i = 0; i < MLP_IN_DIM_PAD * n_points; ++i) {
      float dev = h2f_host(h_pe_fp16[i]);
      if (!compare(ref_pe[i], dev, atol, rtol, (int)i, stage_err, "PE"))
        ++stage_err;
    }
    for (uint32_t i = 0; i < n_points; ++i)
      if (!compare(ref_deltas[i], h_deltas[i], atol, rtol, (int)i, stage_err, "DELTA"))
        ++stage_err;
    if (stage_err) {
      std::cout << "Stage 1 found " << stage_err << " errors, continuing" << std::endl;
      errors += stage_err;
    }
  }

  // ================== Stage 2: MLP (sparse TCU + SIMT act) ==================
  struct LayerParams {
    uint64_t Wc_addr;
    uint64_t meta_addr;
    uint64_t B_addr;
    uint32_t K_in;
    uint32_t N_out;
    uint64_t X_addr;
    uint64_t Y_addr;
    uint64_t act_out_addr;
    const float* W_fp32;  // pruned fp32 weights (for CPU reference)
    const float* B_fp32;
  };
  LayerParams lp[MLP_DEPTH] = {
    { Wc0_addr, M0_addr, B0_addr, MLP_IN_DIM_PAD, MLP_W,
      pe_addr, scratch_fp32_addr, act_a_addr,
      h_W0.data(), h_B0.data() },
    { Wc1_addr, M1_addr, B1_addr, MLP_W, MLP_W,
      act_a_addr, scratch_fp32_addr, act_b_addr,
      h_W1.data(), h_B1.data() },
    { Wc2_addr, M2_addr, B2_addr, MLP_W, MLP_W,
      act_b_addr, scratch_fp32_addr, act_a_addr,
      h_W2.data(), h_B2.data() },
    { Wc3_addr, M3_addr, B3_addr, MLP_W, MLP_OUT_DIM_PAD,
      act_a_addr, out_head_fp32_addr, 0,
      h_W3.data(), h_B3.data() },
  };

  std::vector<float> cpu_pe(MLP_IN_DIM_PAD * n_points);
  std::vector<float> cpu_deltas(n_points);
  ray_setup_cpu(cpu_pe.data(), cpu_deltas.data(),
                h_rays_o.data(), h_rays_d.data(), h_aabb,
                n_rays, n_samples, min_near);
  std::vector<float> cpu_X = cpu_pe;
  std::vector<float> cpu_Y(MLP_W * n_points);

  for (uint32_t L = 0; L < MLP_DEPTH; ++L) {
    const auto& p = lp[L];
    std::cout << "=== Stage 2.L" << L
              << ": MLP GEMM (sparse TCU) " << p.N_out << "x" << p.K_in
              << " * " << p.K_in << "x" << n_points << " ===" << std::endl;

    kernel_arg.W_cur_addr    = p.Wc_addr;
    kernel_arg.meta_cur_addr = p.meta_addr;
    kernel_arg.B_cur_addr    = p.B_addr;
    kernel_arg.X_cur_addr    = p.X_addr;
    kernel_arg.Y_cur_addr    = p.Y_addr;
    kernel_arg.act_out_addr  = p.act_out_addr;
    kernel_arg.K_in_cur      = p.K_in;
    kernel_arg.N_out_cur     = p.N_out;
    kernel_arg.layer_idx     = L;
    kernel_arg.kernel_id     = KID_MLP_GEMM;
    RT_CHECK(vx_copy_to_dev(args_buffer, &kernel_arg, 0, sizeof(kernel_arg_t)));
    {
      uint32_t grid_dim[2] = {n_points / TN, p.N_out / TM};
      RT_CHECK(vx_start_g(device, krnl_buffer, args_buffer, 2, grid_dim, block_dim, 0));
    }
    RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
    {
      char tag[32];
      std::snprintf(tag, sizeof(tag), "MLP_L%u_GEMM", (unsigned)L);
      uint32_t nb = (n_points / TN) * (p.N_out / TM);
      cyc_mlp_gemm += read_back_cycles_tag(tag, nb, true);
    }

    // CPU reference: Y = W_pruned * X
    cpu_Y.assign(p.N_out * n_points, 0.0f);
    layer_gemm_cpu(cpu_Y.data(), p.W_fp32, cpu_X.data(),
                   p.N_out, p.K_in, n_points);

    std::vector<float> dev_Y(p.N_out * n_points);
    uint32_t y_bytes = p.N_out * n_points * sizeof(float);
    RT_CHECK(vx_copy_from_dev(dev_Y.data(),
                              (p.Y_addr == scratch_fp32_addr) ? scratch_fp32_buf : out_head_fp32_buf,
                              0, y_bytes));
    {
      int stage_err = 0;
      for (uint32_t i = 0; i < p.N_out * n_points; ++i)
        if (!compare(cpu_Y[i], dev_Y[i], atol, rtol, (int)i, stage_err, "Y"))
          ++stage_err;
      if (stage_err) {
        std::cout << "Layer " << L << " GEMM found " << stage_err
                  << " errors, continuing" << std::endl;
        errors += stage_err;
      }
    }

    // Activation dispatch
    if (L < MLP_DEPTH - 1) {
      kernel_arg.kernel_id = KID_MLP_ACT;
      RT_CHECK(vx_copy_to_dev(args_buffer, &kernel_arg, 0, sizeof(kernel_arg_t)));
      uint32_t nt_act = p.N_out * n_points;
      uint32_t nb_act = (nt_act + NUM_THREADS - 1) / NUM_THREADS;
      {
        uint32_t grid_dim[2] = {nb_act, 1};
        RT_CHECK(vx_start_g(device, krnl_buffer, args_buffer, 1, grid_dim, block_dim, 0));
      }
      RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
      {
        char tag[32];
        std::snprintf(tag, sizeof(tag), "MLP_L%u_ACT", (unsigned)L);
        cyc_mlp_act += read_back_cycles_tag(tag, nb_act, true);
      }
      cpu_X.assign(p.N_out * n_points, 0.0f);
      for (uint32_t m = 0; m < p.N_out; ++m) {
        float bm = p.B_fp32[m];
        for (uint32_t n = 0; n < n_points; ++n) {
          float v = cpu_Y[m*n_points + n] + bm;
          cpu_X[m*n_points + n] = (v > 0.0f) ? v : 0.0f;
        }
      }
    } else {
      kernel_arg.kernel_id = KID_MLP_OUT_ACT;
      RT_CHECK(vx_copy_to_dev(args_buffer, &kernel_arg, 0, sizeof(kernel_arg_t)));
      uint32_t nb_oa = (n_points + NUM_THREADS - 1) / NUM_THREADS;
      {
        uint32_t grid_dim[2] = {nb_oa, 1};
        RT_CHECK(vx_start_g(device, krnl_buffer, args_buffer, 1, grid_dim, block_dim, 0));
      }
      RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
      cyc_mlp_act += read_back_cycles_tag("MLP_OUT_ACT", nb_oa, true);
    }
  }

  // ---- Verify sigmas + rgbs ----
  std::vector<float> ref_sigmas(n_points);
  std::vector<float> ref_rgbs(n_points * 3);
  for (uint32_t i = 0; i < n_points; ++i) {
    float s = cpu_Y[0*n_points + i] + h_B3[0];
    float r = cpu_Y[1*n_points + i] + h_B3[1];
    float g = cpu_Y[2*n_points + i] + h_B3[2];
    float b = cpu_Y[3*n_points + i] + h_B3[3];
    ref_sigmas[i]      = std::log(1.0f + std::exp(s));
    ref_rgbs[i*3 + 0]  = 1.0f / (1.0f + std::exp(-r));
    ref_rgbs[i*3 + 1]  = 1.0f / (1.0f + std::exp(-g));
    ref_rgbs[i*3 + 2]  = 1.0f / (1.0f + std::exp(-b));
  }
  std::vector<float> h_sigmas(n_points), h_rgbs(n_points * 3);
  RT_CHECK(vx_copy_from_dev(h_sigmas.data(), sigma_buf, 0, sigma_bytes));
  RT_CHECK(vx_copy_from_dev(h_rgbs.data(),   rgb_buf,   0, rgb_bytes));
  {
    int stage_err = 0;
    for (uint32_t i = 0; i < n_points; ++i)
      if (!compare(ref_sigmas[i], h_sigmas[i], atol, rtol, (int)i, stage_err, "SIGMA"))
        ++stage_err;
    for (uint32_t i = 0; i < n_points * 3; ++i)
      if (!compare(ref_rgbs[i], h_rgbs[i], atol, rtol, (int)i, stage_err, "RGB"))
        ++stage_err;
    if (stage_err) {
      std::cout << "Output activation found " << stage_err << " errors" << std::endl;
      errors += stage_err;
    }
  }

  // ================== Stage 3: composite ==================
  std::cout << "=== Stage 3: alpha compositing ===" << std::endl;
  kernel_arg.kernel_id = KID_COMPOSITE;
  RT_CHECK(vx_copy_to_dev(args_buffer, &kernel_arg, 0, sizeof(kernel_arg_t)));
  {
    uint32_t grid_dim[2] = {gd_comp, 1};
    RT_CHECK(vx_start_g(device, krnl_buffer, args_buffer, 1, grid_dim, block_dim, 0));
  }
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
  cyc_comp = read_back_cycles_tag("COMP", gd_comp, true);

  std::vector<float> h_image(n_rays * 3);
  RT_CHECK(vx_copy_from_dev(h_image.data(), image_buf, 0, image_bytes));
  {
    std::vector<float> ref_image(n_rays * 3);
    for (uint32_t ray = 0; ray < n_rays; ++ray) {
      float T = 1.0f, R = 0, G = 0, B = 0;
      uint32_t base = ray * n_samples;
      for (uint32_t s = 0; s < n_samples; ++s) {
        float sigma = ref_sigmas[base + s];
        float delta = cpu_deltas[base + s];
        float alpha = 1.0f - std::exp(-sigma * delta);
        float w = alpha * T;
        R += w * ref_rgbs[(base+s)*3 + 0];
        G += w * ref_rgbs[(base+s)*3 + 1];
        B += w * ref_rgbs[(base+s)*3 + 2];
        T *= (1.0f - alpha);
      }
      ref_image[ray*3 + 0] = R;
      ref_image[ray*3 + 1] = G;
      ref_image[ray*3 + 2] = B;
    }
    int stage_err = 0;
    for (uint32_t i = 0; i < n_rays * 3; ++i)
      if (!compare(ref_image[i], h_image[i], atol, rtol, (int)i, stage_err, "IMAGE"))
        ++stage_err;
    if (stage_err) {
      std::cout << "Stage 3 found " << stage_err << " errors" << std::endl;
      errors += stage_err;
    }
  }

  // ================== Aggregated cycle report ==================
  uint64_t non_gemm = cyc_setup + cyc_mlp_act + cyc_comp;
  uint64_t gemm     = cyc_mlp_gemm;
  uint64_t total    = non_gemm + gemm;
  printf("KCYC[SETUP,nt=%u]: %lu\n",    (unsigned)NUM_THREADS, (unsigned long)cyc_setup);
  printf("KCYC[MLP_GEMM,nt=%u]: %lu\n", (unsigned)NUM_THREADS, (unsigned long)cyc_mlp_gemm);
  printf("KCYC[MLP_ACT,nt=%u]: %lu\n",  (unsigned)NUM_THREADS, (unsigned long)cyc_mlp_act);
  printf("KCYC[COMP,nt=%u]: %lu\n",     (unsigned)NUM_THREADS, (unsigned long)cyc_comp);
  printf("KCYC[SUMMARY,nt=%u]: non_gemm=%lu gemm=%lu total=%lu",
         (unsigned)NUM_THREADS,
         (unsigned long)non_gemm, (unsigned long)gemm, (unsigned long)total);
  if (total > 0) printf("  (gemm_frac=%.3f)", (double)gemm / (double)total);
  printf("\n");

  cleanup();

  if (errors != 0) {
    std::cout << "Found " << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return errors;
  }
  std::cout << "PASSED!" << std::endl;
  return 0;
}
