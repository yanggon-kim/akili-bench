#include <iostream>
#include <unistd.h>
#include <string.h>
#include <cstring>
#include <vector>
#include <chrono>
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
#include <tensor.h>  // prune_2to4_matrix, compress_2to4_matrix
namespace vt = vortex::tensor;
using cfg = vt::wmma_config_t<NUM_TCU_LANES, vt::fp16, vt::fp32>;
static constexpr uint32_t TM = cfg::tileM;
static constexpr uint32_t TN = cfg::tileN;
static constexpr uint32_t TK = cfg::tileK;

// -----------------------------------------------------------------------------
// Pack per-group 2:4 masks into SRAM layout expected by VX_tcu_meta.
// Lifted verbatim from attention/main.cpp — parameterized by (M, K).
// -----------------------------------------------------------------------------
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
  uint32_t KS = K;
  uint32_t num_groups_per_row = KS / 4;
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
     if (0 == _ret)                                             \
       break;                                                   \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
     cleanup();                                                 \
     exit(-1);                                                  \
   } while (false)

// -----------------------------------------------------------------------------
// fp32 ↔ fp16 (same manual IEEE encode as attention/flash).
// -----------------------------------------------------------------------------
static inline uint16_t f2h_host(float x) {
  uint32_t bits;
  std::memcpy(&bits, &x, sizeof(bits));
  uint16_t sign = (bits >> 16) & 0x8000;
  uint16_t mant = (bits >> 13) & 0x03FF;
  int32_t  exp  = (int32_t)((bits >> 23) & 0xFF) - 127 + 15;
  if (exp >= 31) return sign | 0x7C00;
  if (exp <= 0)  return sign;
  return sign | ((uint16_t)exp << 10) | mant;
}
static inline float h2f_host(uint16_t x) {
  uint32_t sign = (uint32_t)(x & 0x8000) << 16;
  uint32_t exp  = (x >> 10) & 0x1F;
  uint32_t mant = (x & 0x3FF);
  uint32_t bits;
  if (exp == 0)       bits = sign;
  else if (exp == 31) bits = sign | 0x7F800000 | (mant << 13);
  else                bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

static float frand() { return 2.0f * (float(rand()) / RAND_MAX) - 1.0f; }

// Host reference: standard conv2d, valid padding, stride 1.
static void conv2d_cpu(std::vector<float>& O,
                       const std::vector<float>& I,
                       const std::vector<float>& W,
                       uint32_t C_in, uint32_t C_out,
                       uint32_t H, uint32_t Wd, uint32_t K,
                       uint32_t H_out, uint32_t W_out) {
  O.assign((size_t)C_out * H_out * W_out, 0.0f);
  for (uint32_t oc = 0; oc < C_out; ++oc) {
    for (uint32_t oy = 0; oy < H_out; ++oy) {
      for (uint32_t ox = 0; ox < W_out; ++ox) {
        double sum = 0.0;
        for (uint32_t ic = 0; ic < C_in; ++ic) {
          for (uint32_t ky = 0; ky < K; ++ky) {
            for (uint32_t kx = 0; kx < K; ++kx) {
              uint32_t in_idx = ic * H * Wd + (oy + ky) * Wd + (ox + kx);
              uint32_t wt_idx = ((oc * C_in + ic) * K + ky) * K + kx;
              sum += (double)I[in_idx] * (double)W[wt_idx];
            }
          }
        }
        O[oc * H_out * W_out + oy * W_out + ox] = (float)sum;
      }
    }
  }
}

// -----------------------------------------------------------------------------
// Weight flattening: W[C_out × C_in × K × K] row-major → W_gemm[M_gemm × K_gemm]
// with zero padding to M_gemm and K_gemm (next multiples of TM / TK).
// -----------------------------------------------------------------------------
static void flatten_weights_gemm(std::vector<float>& W_gemm,
                                 const std::vector<float>& W,
                                 uint32_t C_in, uint32_t C_out,
                                 uint32_t K,
                                 uint32_t M_gemm, uint32_t K_gemm) {
  W_gemm.assign((size_t)M_gemm * K_gemm, 0.0f);
  uint32_t KK = C_in * K * K;
  for (uint32_t oc = 0; oc < C_out; ++oc) {
    for (uint32_t k = 0; k < KK; ++k) {
      W_gemm[oc * K_gemm + k] = W[oc * KK + k];
    }
  }
}

// -----------------------------------------------------------------------------
// im2col for B matrix.  Output: Icol [N_gemm × K_gemm] row-major. Each row is
// one output spatial location's flattened receptive field; padded with zeros
// to K_gemm and out to N_gemm.
// This layout is col-major B for sgemm_tcu's load_matrix_sync<col_major>.
// -----------------------------------------------------------------------------
static void im2col_cpu(std::vector<float>& Icol,
                       const std::vector<float>& I,
                       uint32_t C_in, uint32_t H, uint32_t Wd,
                       uint32_t K, uint32_t H_out, uint32_t W_out,
                       uint32_t N_gemm, uint32_t K_gemm) {
  Icol.assign((size_t)N_gemm * K_gemm, 0.0f);
  for (uint32_t oy = 0; oy < H_out; ++oy) {
    for (uint32_t ox = 0; ox < W_out; ++ox) {
      uint32_t out_idx = oy * W_out + ox;
      uint32_t col_base = out_idx * K_gemm;
      uint32_t k_idx = 0;
      for (uint32_t ic = 0; ic < C_in; ++ic) {
        for (uint32_t ky = 0; ky < K; ++ky) {
          for (uint32_t kx = 0; kx < K; ++kx) {
            uint32_t in_idx = ic * H * Wd + (oy + ky) * Wd + (ox + kx);
            Icol[col_base + k_idx++] = I[in_idx];
          }
        }
      }
      // padding to K_gemm is zero (left by assign)
    }
  }
}

// -----------------------------------------------------------------------------
// Shape parameters & state
// -----------------------------------------------------------------------------
const char* kernel_file = "kernel.vxbin";
uint32_t C_in_req  = 1;
uint32_t C_out_req = 8;
uint32_t H_req     = 28;
uint32_t W_req     = 28;
uint32_t K_req     = 3;
uint32_t mode      = MODE_SIMT;

vx_device_h device = nullptr;
vx_buffer_h I_buffer = nullptr;
vx_buffer_h O_buffer = nullptr;
vx_buffer_h W_fp32_buffer = nullptr;  // SIMT path: fp32 W
vx_buffer_h W_fp16_buffer = nullptr;  // TCU dense path: fp16 W_gemm (M_gemm × K_gemm)
vx_buffer_h Wsp_buffer    = nullptr;  // TCU sparse path: compressed fp16 W
vx_buffer_h meta_W_buffer = nullptr;  // TCU sparse path: packed metadata
vx_buffer_h Icol_buffer   = nullptr;  // TCU paths: im2col fp16 [N_gemm × K_gemm]
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
  std::cout << "conv_tcu — im2col+GEMM convolution benchmark (SIMT / dense TCU / sparse TCU)" << std::endl;
  std::cout << "Usage: [-c C_in] [-o C_out] [-h H] [-w W] [-s K_size] [-t mode 0/1/2]" << std::endl;
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "c:o:h:w:s:t:k:?")) != -1) {
    switch (c) {
      case 'c': C_in_req  = atoi(optarg); break;
      case 'o': C_out_req = atoi(optarg); break;
      case 'h': H_req     = atoi(optarg); break;
      case 'w': W_req     = atoi(optarg); break;
      case 's': K_req     = atoi(optarg); break;
      case 't': mode      = atoi(optarg); break;
      case 'k': kernel_file = optarg; break;
      default: show_usage(); exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    if (I_buffer)      vx_mem_free(I_buffer);
    if (O_buffer)      vx_mem_free(O_buffer);
    if (W_fp32_buffer) vx_mem_free(W_fp32_buffer);
    if (W_fp16_buffer) vx_mem_free(W_fp16_buffer);
    if (Wsp_buffer)    vx_mem_free(Wsp_buffer);
    if (meta_W_buffer) vx_mem_free(meta_W_buffer);
    if (Icol_buffer)   vx_mem_free(Icol_buffer);
    if (krnl_buffer)   vx_mem_free(krnl_buffer);
    if (args_buffer)   vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

static void read_back_cycles(const char* stage_tag) {
  kernel_arg_t back = {};
  vx_copy_from_dev(&back, args_buffer, 0, sizeof(kernel_arg_t));
  printf("KCYC[%s,mode=%u,nt=%u]: %lu\n",
         stage_tag, (unsigned)mode, (unsigned)NUM_THREADS,
         (unsigned long)back.kernel_cycles);
}

int main(int argc, char* argv[]) {
  parse_args(argc, argv);
  std::srand(50);

  RT_CHECK(vx_dev_open(&device));
  uint64_t NT_caps = 0;
  vx_dev_caps(device, VX_CAPS_NUM_THREADS, &NT_caps);
  if ((uint32_t)NT_caps != NUM_THREADS) {
    printf("Error: device NUM_THREADS=%u but benchmark built with NUM_THREADS=%u\n",
           (unsigned)NT_caps, (unsigned)NUM_THREADS);
    cleanup();
    return -1;
  }

  uint32_t C_in  = C_in_req;
  uint32_t C_out = C_out_req;
  uint32_t H     = H_req;
  uint32_t Wd    = W_req;
  uint32_t K     = K_req;
  uint32_t H_out = H - K + 1;
  uint32_t W_out = Wd - K + 1;
  uint32_t N_out = H_out * W_out;
  uint32_t KK    = C_in * K * K;

  // Pad GEMM dims to tile multiples.
  auto round_up = [](uint32_t v, uint32_t m) { return ((v + m - 1) / m) * m; };
  uint32_t M_gemm = round_up(C_out, TM);
  uint32_t N_gemm = round_up(N_out, TN);
  uint32_t K_gemm = round_up(KK, TK);
  // sparse requires K % 4 == 0 — already satisfied by TK (16 for fp16) which is a multiple of 4.

  std::cout << "conv_tcu — C_in=" << C_in << " C_out=" << C_out
            << " H=" << H << " W=" << Wd << " K=" << K
            << " → H_out=" << H_out << " W_out=" << W_out
            << "  (GEMM M=" << M_gemm << " N=" << N_gemm << " K=" << K_gemm << ")"
            << "  TM=" << TM << " TN=" << TN << " TK=" << TK << std::endl;

  // -------------------------------------------------------------------
  // Generate random input + weights (fp32 host reference ground truth).
  // -------------------------------------------------------------------
  std::vector<float> h_I((size_t)C_in * H * Wd);
  std::vector<float> h_W((size_t)C_out * KK);
  for (auto& v : h_I) v = frand();
  for (auto& v : h_W) v = frand();

  std::vector<float> h_W_gemm;
  flatten_weights_gemm(h_W_gemm, h_W, C_in, C_out, K, M_gemm, K_gemm);

  // -------------------------------------------------------------------
  // Compute CPU reference before any in-place pruning touches h_W.
  // -------------------------------------------------------------------
  std::vector<float> h_O_ref;
  conv2d_cpu(h_O_ref, h_I, h_W, C_in, C_out, H, Wd, K, H_out, W_out);

  // -------------------------------------------------------------------
  // Device buffers
  // -------------------------------------------------------------------
  // SIMT conv reads fp32 W directly from W_fp32_buffer. GEMM output buffer is
  // [M_gemm × N_gemm] fp32 for TCU paths, but SIMT writes [C_out × H_out × W_out]
  // directly to a compact layout. We allocate the larger of the two to keep one
  // buffer across modes.
  size_t simt_O_floats = (size_t)C_out * H_out * W_out;
  size_t gemm_O_floats = (size_t)M_gemm * N_gemm;
  size_t O_floats = std::max(simt_O_floats, gemm_O_floats);
  size_t O_bytes = O_floats * sizeof(float);

  RT_CHECK(vx_mem_alloc(device, h_I.size() * sizeof(float), VX_MEM_READ_WRITE, &I_buffer));
  RT_CHECK(vx_mem_address(I_buffer, &kernel_arg.I_addr));
  RT_CHECK(vx_mem_alloc(device, O_bytes, VX_MEM_READ_WRITE, &O_buffer));
  RT_CHECK(vx_mem_address(O_buffer, &kernel_arg.O_addr));

  RT_CHECK(vx_copy_to_dev(I_buffer, h_I.data(), 0, h_I.size() * sizeof(float)));

  // SIMT path: upload raw fp32 W.
  if (mode == MODE_SIMT) {
    RT_CHECK(vx_mem_alloc(device, h_W.size() * sizeof(float), VX_MEM_READ_WRITE, &W_fp32_buffer));
    RT_CHECK(vx_mem_address(W_fp32_buffer, &kernel_arg.W_addr));
    RT_CHECK(vx_copy_to_dev(W_fp32_buffer, h_W.data(), 0, h_W.size() * sizeof(float)));
  }

  // Dense TCU path: convert W_gemm and im2col(I) to fp16 and upload.
  if (mode == MODE_DENSE_TCU) {
    std::vector<uint16_t> h_W_fp16((size_t)M_gemm * K_gemm);
    for (size_t i = 0; i < h_W_fp16.size(); ++i) h_W_fp16[i] = f2h_host(h_W_gemm[i]);
    RT_CHECK(vx_mem_alloc(device, h_W_fp16.size() * sizeof(uint16_t),
                          VX_MEM_READ_WRITE, &W_fp16_buffer));
    RT_CHECK(vx_mem_address(W_fp16_buffer, &kernel_arg.W_addr));
    RT_CHECK(vx_copy_to_dev(W_fp16_buffer, h_W_fp16.data(), 0,
                            h_W_fp16.size() * sizeof(uint16_t)));

    std::vector<float> h_Icol_fp32;
    im2col_cpu(h_Icol_fp32, h_I, C_in, H, Wd, K, H_out, W_out, N_gemm, K_gemm);
    std::vector<uint16_t> h_Icol_fp16((size_t)N_gemm * K_gemm);
    for (size_t i = 0; i < h_Icol_fp16.size(); ++i) h_Icol_fp16[i] = f2h_host(h_Icol_fp32[i]);
    RT_CHECK(vx_mem_alloc(device, h_Icol_fp16.size() * sizeof(uint16_t),
                          VX_MEM_READ_WRITE, &Icol_buffer));
    RT_CHECK(vx_mem_address(Icol_buffer, &kernel_arg.B_addr));
    RT_CHECK(vx_copy_to_dev(Icol_buffer, h_Icol_fp16.data(), 0,
                            h_Icol_fp16.size() * sizeof(uint16_t)));
  }

  // Sparse TCU path: prune W_gemm 2:4, compress, pack metadata, upload both W
  // (compressed fp16, half K-stride) and the im2col fp16 B (same as dense).
  if (mode == MODE_SPARSE_TCU) {
    std::vector<uint16_t> h_W_fp16((size_t)M_gemm * K_gemm);
    for (size_t i = 0; i < h_W_fp16.size(); ++i) h_W_fp16[i] = f2h_host(h_W_gemm[i]);
    if (!vt::prune_2to4_matrix<vt::fp16>(h_W_fp16.data(), M_gemm, K_gemm)) {
      printf("Error: prune_2to4_matrix failed for W (M=%u, K=%u)\n", M_gemm, K_gemm);
      cleanup();
      return -1;
    }

    // Re-materialize pruned W_gemm in fp32 and recompute reference so the
    // comparison tolerates the pruning approximation honestly.
    for (size_t i = 0; i < h_W_gemm.size(); ++i) h_W_gemm[i] = h2f_host(h_W_fp16[i]);
    // Unflatten back into h_W (original layout) for the CPU reference.
    for (uint32_t oc = 0; oc < C_out; ++oc) {
      for (uint32_t k = 0; k < KK; ++k) {
        h_W[oc * KK + k] = h_W_gemm[oc * K_gemm + k];
      }
    }
    conv2d_cpu(h_O_ref, h_I, h_W, C_in, C_out, H, Wd, K, H_out, W_out);

    // Compress + pack metadata
    std::vector<uint16_t> h_W_compressed((size_t)M_gemm * (K_gemm / 2));
    std::vector<uint8_t>  sparse_masks;
    if (!vt::compress_2to4_matrix<vt::fp16>(h_W_compressed.data(),
                                            h_W_fp16.data(),
                                            sparse_masks, M_gemm, K_gemm)) {
      printf("Error: compress_2to4_matrix failed for W\n");
      cleanup();
      return -1;
    }
    std::vector<uint32_t> h_meta_W;
    pack_metadata(h_meta_W, sparse_masks, M_gemm, K_gemm);

    uint32_t wsp_bytes  = (uint32_t)h_W_compressed.size() * sizeof(uint16_t);
    uint32_t wmeta_bytes = (uint32_t)h_meta_W.size() * sizeof(uint32_t);
    RT_CHECK(vx_mem_alloc(device, wsp_bytes, VX_MEM_READ_WRITE, &Wsp_buffer));
    RT_CHECK(vx_mem_address(Wsp_buffer, &kernel_arg.W_addr));
    RT_CHECK(vx_mem_alloc(device, wmeta_bytes, VX_MEM_READ_WRITE, &meta_W_buffer));
    RT_CHECK(vx_mem_address(meta_W_buffer, &kernel_arg.meta_W_addr));
    RT_CHECK(vx_copy_to_dev(Wsp_buffer, h_W_compressed.data(), 0, wsp_bytes));
    RT_CHECK(vx_copy_to_dev(meta_W_buffer, h_meta_W.data(), 0, wmeta_bytes));

    // im2col B (same as dense)
    std::vector<float> h_Icol_fp32;
    im2col_cpu(h_Icol_fp32, h_I, C_in, H, Wd, K, H_out, W_out, N_gemm, K_gemm);
    std::vector<uint16_t> h_Icol_fp16((size_t)N_gemm * K_gemm);
    for (size_t i = 0; i < h_Icol_fp16.size(); ++i) h_Icol_fp16[i] = f2h_host(h_Icol_fp32[i]);
    RT_CHECK(vx_mem_alloc(device, h_Icol_fp16.size() * sizeof(uint16_t),
                          VX_MEM_READ_WRITE, &Icol_buffer));
    RT_CHECK(vx_mem_address(Icol_buffer, &kernel_arg.B_addr));
    RT_CHECK(vx_copy_to_dev(Icol_buffer, h_Icol_fp16.data(), 0,
                            h_Icol_fp16.size() * sizeof(uint16_t)));
  }

  // upload kernel binary
  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));

  // -------------------------------------------------------------------
  // Dispatch
  // -------------------------------------------------------------------
  kernel_arg.C_in  = C_in;
  kernel_arg.C_out = C_out;
  kernel_arg.H     = H;
  kernel_arg.W     = Wd;
  kernel_arg.K_sz  = K;
  kernel_arg.H_out = H_out;
  kernel_arg.W_out = W_out;
  kernel_arg.M_gemm = M_gemm;
  kernel_arg.N_gemm = N_gemm;
  kernel_arg.K_gemm = K_gemm;

  if (mode == MODE_SIMT) {
    // grid = (H_out*W_out, C_out)
    kernel_arg.grid_dim[0] = N_out;
    kernel_arg.grid_dim[1] = C_out;
    kernel_arg.block_dim[0] = 0;
    kernel_arg.block_dim[1] = 0;
    kernel_arg.kernel_id = KID_CONV_SIMT;
  } else if (mode == MODE_DENSE_TCU) {
    kernel_arg.grid_dim[0] = N_gemm / TN;   // blockIdx.x = tile_col
    kernel_arg.grid_dim[1] = M_gemm / TM;   // blockIdx.y = tile_row
    kernel_arg.block_dim[0] = NUM_THREADS;
    kernel_arg.block_dim[1] = 1;
    kernel_arg.kernel_id = KID_CONV_TCU;
  } else {
    kernel_arg.grid_dim[0] = N_gemm / TN;
    kernel_arg.grid_dim[1] = M_gemm / TM;
    kernel_arg.block_dim[0] = NUM_THREADS;
    kernel_arg.block_dim[1] = 1;
    kernel_arg.kernel_id = KID_CONV_SPARSE;
  }

  RT_CHECK(vx_mem_alloc(device, sizeof(kernel_arg_t), VX_MEM_READ_WRITE, &args_buffer));
  RT_CHECK(vx_copy_to_dev(args_buffer, &kernel_arg, 0, sizeof(kernel_arg_t)));
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
  read_back_cycles("CONV");

  // -------------------------------------------------------------------
  // Download output
  // -------------------------------------------------------------------
  std::vector<float> h_O(C_out * H_out * W_out, 0.0f);
  if (mode == MODE_SIMT) {
    // Kernel writes directly into compact [C_out × H_out × W_out] layout.
    RT_CHECK(vx_copy_from_dev(h_O.data(), O_buffer, 0, h_O.size() * sizeof(float)));
  } else {
    // TCU kernels write into padded [M_gemm × N_gemm] layout.  Extract the
    // valid (oc < C_out, (oy, ox) valid) region.
    std::vector<float> h_O_gemm((size_t)M_gemm * N_gemm, 0.0f);
    RT_CHECK(vx_copy_from_dev(h_O_gemm.data(), O_buffer, 0, h_O_gemm.size() * sizeof(float)));
    for (uint32_t oc = 0; oc < C_out; ++oc) {
      for (uint32_t oy = 0; oy < H_out; ++oy) {
        for (uint32_t ox = 0; ox < W_out; ++ox) {
          h_O[oc * H_out * W_out + oy * W_out + ox] =
              h_O_gemm[oc * N_gemm + (oy * W_out + ox)];
        }
      }
    }
  }

  // -------------------------------------------------------------------
  // Verify — compare to CPU reference (or pruned reference for sparse).
  // -------------------------------------------------------------------
  int errors = 0;
  const float atol = (mode == MODE_SIMT) ? 1e-3f : 5e-2f;
  const float rtol = (mode == MODE_SIMT) ? 1e-3f : 5e-2f;
  for (size_t i = 0; i < h_O_ref.size(); ++i) {
    float a = h_O_ref[i];
    float b = h_O[i];
    float diff = std::fabs(a - b);
    float lim  = atol + rtol * std::fabs(a);
    if (diff > lim) {
      if (errors < 10) {
        printf("*** O error: [%zu] expected=%f actual=%f diff=%f\n",
               i, a, b, diff);
      }
      ++errors;
    }
  }

  cleanup();

  if (errors != 0) {
    printf("Found %d / %zu errors!\n", errors, h_O_ref.size());
    std::cout << "FAILED!" << std::endl;
    return errors;
  }
  std::cout << "PASSED!" << std::endl;
  return 0;
}
