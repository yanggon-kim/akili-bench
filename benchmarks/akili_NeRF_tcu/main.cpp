// akili_NeRF_tcu — Full NeRF forward pass with the MLP on the dense TCU.
//
//   Stage 1 (SIMT):  ray-AABB slab + stratified sampling + positional encoding
//   Stage 2 (TCU):   4 MLP layer GEMMs (W x X -> Y_fp32)
//        interleaved with
//   Stage 2 (SIMT):  ReLU/softplus/sigmoid + bias + fp32->fp16 cast
//   Stage 3 (SIMT):  per-ray alpha compositing
//
// Ray setup + compositing + activations run as the SIMT baseline does;
// only the 4 GEMMs move to the TCU.

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
namespace vt = vortex::tensor;
using cfg = vt::wmma_config_t<NUM_TCU_LANES, vt::fp16, vt::fp32>;
static constexpr uint32_t TM = cfg::tileM;
static constexpr uint32_t TN = cfg::tileN;
static constexpr uint32_t TK = cfg::tileK;

#define RT_CHECK(_expr)                                         \
   do {                                                         \
     int _ret = _expr;                                          \
     if (0 == _ret) break;                                      \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
     cleanup();                                                 \
     exit(-1);                                                  \
   } while (false)

// ---------------------------------------------------------------------------
// Host-side fp32 <-> fp16 (IEEE) packing.
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

// ---------------------------------------------------------------------------
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
// CPU reference — mirrors the device math.  Uses fp32 throughout so we can
// compare the device result (which used fp16 operands) at atol=rtol=1e-2.
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

// Generic layer GEMM in fp32:  C[N_out x n_pts] = W[N_out x K_in] * X[K_in x n_pts]
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
vx_buffer_h W0_buf = nullptr, W1_buf = nullptr, W2_buf = nullptr, W3_buf = nullptr;
vx_buffer_h B0_buf = nullptr, B1_buf = nullptr, B2_buf = nullptr, B3_buf = nullptr;
vx_buffer_h act_a_buf = nullptr, act_b_buf = nullptr;
vx_buffer_h scratch_fp32_buf = nullptr;
vx_buffer_h out_head_fp32_buf = nullptr;
vx_buffer_h sigma_buf = nullptr, rgb_buf = nullptr, image_buf = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

uint64_t W0_addr=0, W1_addr=0, W2_addr=0, W3_addr=0;
uint64_t B0_addr=0, B1_addr=0, B2_addr=0, B3_addr=0;
uint64_t pe_addr=0, act_a_addr=0, act_b_addr=0;
uint64_t scratch_fp32_addr=0, out_head_fp32_addr=0;

static void show_usage() {
  std::cout << "akili_NeRF_tcu — Dense TCU full-NeRF forward" << std::endl;
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
    if (W0_buf) vx_mem_free(W0_buf);
    if (W1_buf) vx_mem_free(W1_buf);
    if (W2_buf) vx_mem_free(W2_buf);
    if (W3_buf) vx_mem_free(W3_buf);
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
    if (krnl_buffer) vx_mem_free(krnl_buffer);
    if (args_buffer) vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

static uint64_t read_back_cycles_tag(const char* tag, bool print) {
  kernel_arg_t back = {};
  vx_copy_from_dev(&back, args_buffer, 0, sizeof(kernel_arg_t));
  if (print)
    printf("KCYC[%s,nt=%u]: %lu\n",
           tag, (unsigned)NUM_THREADS, (unsigned long)back.kernel_cycles);
  return (uint64_t)back.kernel_cycles;
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

  // Require that n_rays * n_samples is a multiple of tileN = 8 and each
  // N_out is a multiple of tileM = 8 (enforced by MLP_W=64 and
  // MLP_OUT_DIM_PAD=8 at compile time).
  uint32_t n_rays    = n_rays_req;
  uint32_t n_samples = n_samples_req;
  uint32_t n_points  = n_rays * n_samples;
  if (n_points % TN != 0) {
    printf("Error: n_rays*n_samples=%u must be a multiple of TN=%u\n",
           n_points, TN);
    cleanup();
    return -1;
  }
  std::cout << "akili_NeRF_tcu — n_rays=" << n_rays
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
  // Layer 0: [MLP_W x MLP_IN_DIM_PAD] row-major
  // Layer 1: [MLP_W x MLP_W]
  // Layer 2: [MLP_W x MLP_W]
  // Layer 3: [MLP_OUT_DIM_PAD x MLP_W]  (rows 0..3 are real, 4..7 zero)
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
  // layer 3: only rows 0..3 active (rest stay zero)
  for (uint32_t r = 0; r < MLP_OUT_DIM; ++r)
    for (uint32_t k = 0; k < MLP_W; ++k)
      h_W3[r * MLP_W + k] = urnd();

  std::vector<float> h_B0(MLP_W, 0.0f);
  std::vector<float> h_B1(MLP_W, 0.0f);
  std::vector<float> h_B2(MLP_W, 0.0f);
  std::vector<float> h_B3(MLP_OUT_DIM_PAD, 0.0f);
  h_B3[0] = 0.5f;

  // ---- Convert weights to fp16 for device upload ----
  std::vector<uint16_t> h_W0_fp16(h_W0.size());
  std::vector<uint16_t> h_W1_fp16(h_W1.size());
  std::vector<uint16_t> h_W2_fp16(h_W2.size());
  std::vector<uint16_t> h_W3_fp16(h_W3.size());
  for (size_t i = 0; i < h_W0.size(); ++i) h_W0_fp16[i] = f2h_host(h_W0[i]);
  for (size_t i = 0; i < h_W1.size(); ++i) h_W1_fp16[i] = f2h_host(h_W1[i]);
  for (size_t i = 0; i < h_W2.size(); ++i) h_W2_fp16[i] = f2h_host(h_W2[i]);
  for (size_t i = 0; i < h_W3.size(); ++i) h_W3_fp16[i] = f2h_host(h_W3[i]);

  // ---- Allocate device buffers ----
  uint32_t rays_bytes       = n_rays * 3 * sizeof(float);
  uint32_t aabb_bytes       = 6 * sizeof(float);
  uint32_t pe_bytes         = MLP_IN_DIM_PAD * n_points * sizeof(uint16_t);
  uint32_t delta_bytes      = n_points * sizeof(float);
  uint32_t act_bytes        = MLP_W * n_points * sizeof(uint16_t);
  uint32_t scratch_bytes    = MLP_W * n_points * sizeof(float);
  uint32_t out_head_bytes   = MLP_OUT_DIM_PAD * n_points * sizeof(float);
  uint32_t sigma_bytes      = n_points * sizeof(float);
  uint32_t rgb_bytes        = n_points * 3 * sizeof(float);
  uint32_t image_bytes      = n_rays * 3 * sizeof(float);

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

  RT_CHECK(vx_mem_alloc(device, h_W0_fp16.size()*sizeof(uint16_t), VX_MEM_READ_WRITE, &W0_buf));
  RT_CHECK(vx_mem_address(W0_buf, &W0_addr));
  RT_CHECK(vx_mem_alloc(device, h_W1_fp16.size()*sizeof(uint16_t), VX_MEM_READ_WRITE, &W1_buf));
  RT_CHECK(vx_mem_address(W1_buf, &W1_addr));
  RT_CHECK(vx_mem_alloc(device, h_W2_fp16.size()*sizeof(uint16_t), VX_MEM_READ_WRITE, &W2_buf));
  RT_CHECK(vx_mem_address(W2_buf, &W2_addr));
  RT_CHECK(vx_mem_alloc(device, h_W3_fp16.size()*sizeof(uint16_t), VX_MEM_READ_WRITE, &W3_buf));
  RT_CHECK(vx_mem_address(W3_buf, &W3_addr));

  RT_CHECK(vx_mem_alloc(device, h_B0.size()*sizeof(float), VX_MEM_READ_WRITE, &B0_buf));
  RT_CHECK(vx_mem_address(B0_buf, &B0_addr));
  RT_CHECK(vx_mem_alloc(device, h_B1.size()*sizeof(float), VX_MEM_READ_WRITE, &B1_buf));
  RT_CHECK(vx_mem_address(B1_buf, &B1_addr));
  RT_CHECK(vx_mem_alloc(device, h_B2.size()*sizeof(float), VX_MEM_READ_WRITE, &B2_buf));
  RT_CHECK(vx_mem_address(B2_buf, &B2_addr));
  RT_CHECK(vx_mem_alloc(device, h_B3.size()*sizeof(float), VX_MEM_READ_WRITE, &B3_buf));
  RT_CHECK(vx_mem_address(B3_buf, &B3_addr));

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

  RT_CHECK(vx_copy_to_dev(W0_buf, h_W0_fp16.data(), 0, h_W0_fp16.size()*sizeof(uint16_t)));
  RT_CHECK(vx_copy_to_dev(W1_buf, h_W1_fp16.data(), 0, h_W1_fp16.size()*sizeof(uint16_t)));
  RT_CHECK(vx_copy_to_dev(W2_buf, h_W2_fp16.data(), 0, h_W2_fp16.size()*sizeof(uint16_t)));
  RT_CHECK(vx_copy_to_dev(W3_buf, h_W3_fp16.data(), 0, h_W3_fp16.size()*sizeof(uint16_t)));
  RT_CHECK(vx_copy_to_dev(B0_buf, h_B0.data(), 0, h_B0.size()*sizeof(float)));
  RT_CHECK(vx_copy_to_dev(B1_buf, h_B1.data(), 0, h_B1.size()*sizeof(float)));
  RT_CHECK(vx_copy_to_dev(B2_buf, h_B2.data(), 0, h_B2.size()*sizeof(float)));
  RT_CHECK(vx_copy_to_dev(B3_buf, h_B3.data(), 0, h_B3.size()*sizeof(float)));

  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
  RT_CHECK(vx_mem_alloc(device, sizeof(kernel_arg_t), VX_MEM_READ_WRITE, &args_buffer));

  int errors = 0;
  const float atol = 1e-2f;
  const float rtol = 1e-2f;

  uint64_t cyc_setup = 0, cyc_mlp_gemm = 0, cyc_mlp_act = 0, cyc_comp = 0;

  // ================== Stage 1: ray setup ==================
  std::cout << "=== Stage 1: ray setup + positional encoding ===" << std::endl;
  kernel_arg.kernel_id = KID_RAY_SETUP;
  RT_CHECK(vx_copy_to_dev(args_buffer, &kernel_arg, 0, sizeof(kernel_arg_t)));
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
  cyc_setup = read_back_cycles_tag("SETUP", true);

  // Readback PE features + deltas and verify against CPU
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

  // ================== Stage 2: MLP — 4 layers ==================
  // Layer params: (W, B, K_in, N_out, X, Y, act_out).
  // For the CPU reference we use the host fp32 weights (which are what
  // the fp16 device weights approximate).
  struct LayerParams {
    uint64_t W_addr;
    uint64_t B_addr;
    uint32_t K_in;
    uint32_t N_out;
    uint64_t X_addr;
    uint64_t Y_addr;
    uint64_t act_out_addr;
    uint64_t act_out_bytes;
    const float* W_fp32;
    const float* B_fp32;
  };
  LayerParams lp[MLP_DEPTH] = {
    { W0_addr, B0_addr, MLP_IN_DIM_PAD, MLP_W,
      pe_addr, scratch_fp32_addr, act_a_addr, (uint64_t)act_bytes,
      h_W0.data(), h_B0.data() },
    { W1_addr, B1_addr, MLP_W, MLP_W,
      act_a_addr, scratch_fp32_addr, act_b_addr, (uint64_t)act_bytes,
      h_W1.data(), h_B1.data() },
    { W2_addr, B2_addr, MLP_W, MLP_W,
      act_b_addr, scratch_fp32_addr, act_a_addr, (uint64_t)act_bytes,
      h_W2.data(), h_B2.data() },
    { W3_addr, B3_addr, MLP_W, MLP_OUT_DIM_PAD,
      act_a_addr, out_head_fp32_addr, 0, 0,
      h_W3.data(), h_B3.data() },
  };

  // CPU reference pipeline (fp32, starts from the CPU-computed PE)
  std::vector<float> cpu_pe(MLP_IN_DIM_PAD * n_points);
  std::vector<float> cpu_deltas(n_points);
  ray_setup_cpu(cpu_pe.data(), cpu_deltas.data(),
                h_rays_o.data(), h_rays_d.data(), h_aabb,
                n_rays, n_samples, min_near);
  std::vector<float> cpu_X = cpu_pe;  // layer 0 input
  std::vector<float> cpu_Y(MLP_W * n_points);
  std::vector<float> cpu_act(MLP_W * n_points);

  for (uint32_t L = 0; L < MLP_DEPTH; ++L) {
    const auto& p = lp[L];
    std::cout << "=== Stage 2.L" << L
              << ": MLP GEMM (TCU) " << p.N_out << "x" << p.K_in
              << " * " << p.K_in << "x" << n_points << " ===" << std::endl;

    kernel_arg.W_cur_addr   = p.W_addr;
    kernel_arg.B_cur_addr   = p.B_addr;
    kernel_arg.X_cur_addr   = p.X_addr;
    kernel_arg.Y_cur_addr   = p.Y_addr;
    kernel_arg.act_out_addr = p.act_out_addr;
    kernel_arg.K_in_cur     = p.K_in;
    kernel_arg.N_out_cur    = p.N_out;
    kernel_arg.layer_idx    = L;
    kernel_arg.grid_dim[0]  = n_points / TN;
    kernel_arg.grid_dim[1]  = p.N_out / TM;
    kernel_arg.block_dim[0] = NUM_THREADS;
    kernel_arg.block_dim[1] = 1;
    kernel_arg.kernel_id    = KID_MLP_GEMM;
    RT_CHECK(vx_copy_to_dev(args_buffer, &kernel_arg, 0, sizeof(kernel_arg_t)));
    RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
    RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
    {
      char tag[32];
      std::snprintf(tag, sizeof(tag), "MLP_L%u_GEMM", (unsigned)L);
      cyc_mlp_gemm += read_back_cycles_tag(tag, true);
    }

    // CPU reference: Y = W * X (fp32)
    cpu_Y.assign(p.N_out * n_points, 0.0f);
    layer_gemm_cpu(cpu_Y.data(), p.W_fp32, cpu_X.data(),
                   p.N_out, p.K_in, n_points);

    // Compare fp32 output
    std::vector<float> dev_Y(p.N_out * n_points);
    uint32_t y_bytes = p.N_out * n_points * sizeof(float);
    RT_CHECK(vx_copy_from_dev(dev_Y.data(),
                              (p.Y_addr == scratch_fp32_addr) ? scratch_fp32_buf : out_head_fp32_buf,
                              0, y_bytes));
    {
      int stage_err = 0;
      for (uint32_t i = 0; i < p.N_out * n_points; ++i) {
        // Layer 3: only rows 0..MLP_OUT_DIM-1 are meaningful; rows 4..7 are
        // zero weights so their outputs are 0 — but the CPU reference also
        // sees zero weights there (we zero-filled h_W3). Comparing all
        // rows is fine.
        if (!compare(cpu_Y[i], dev_Y[i], atol, rtol, (int)i, stage_err, "Y"))
          ++stage_err;
      }
      if (stage_err) {
        std::cout << "Layer " << L << " GEMM found " << stage_err
                  << " errors, continuing" << std::endl;
        errors += stage_err;
      }
    }

    // Dispatch activation kernel
    if (L < MLP_DEPTH - 1) {
      // Hidden layer: ReLU + bias + fp16 cast
      kernel_arg.kernel_id = KID_MLP_ACT;
      RT_CHECK(vx_copy_to_dev(args_buffer, &kernel_arg, 0, sizeof(kernel_arg_t)));
      RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
      RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
      {
        char tag[32];
        std::snprintf(tag, sizeof(tag), "MLP_L%u_ACT", (unsigned)L);
        cyc_mlp_act += read_back_cycles_tag(tag, true);
      }
      // Update cpu_X = ReLU(Y + B)
      cpu_X.assign(p.N_out * n_points, 0.0f);
      for (uint32_t m = 0; m < p.N_out; ++m) {
        float bm = p.B_fp32[m];
        for (uint32_t n = 0; n < n_points; ++n) {
          float v = cpu_Y[m*n_points + n] + bm;
          cpu_X[m*n_points + n] = (v > 0.0f) ? v : 0.0f;
        }
      }
    } else {
      // Layer 3: softplus/sigmoid → sigmas/rgbs
      kernel_arg.kernel_id = KID_MLP_OUT_ACT;
      RT_CHECK(vx_copy_to_dev(args_buffer, &kernel_arg, 0, sizeof(kernel_arg_t)));
      RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
      RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
      cyc_mlp_act += read_back_cycles_tag("MLP_OUT_ACT", true);
    }
  }

  // ---- Verify sigmas + rgbs against CPU reference (using cpu_Y from layer 3) ----
  std::vector<float> ref_sigmas(n_points);
  std::vector<float> ref_rgbs(n_points * 3);
  // cpu_Y currently holds the layer-3 fp32 output (before activation)
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
      std::cout << "Output activation found " << stage_err
                << " errors, continuing" << std::endl;
      errors += stage_err;
    }
  }

  // ================== Stage 3: composite ==================
  std::cout << "=== Stage 3: alpha compositing ===" << std::endl;
  kernel_arg.kernel_id = KID_COMPOSITE;
  RT_CHECK(vx_copy_to_dev(args_buffer, &kernel_arg, 0, sizeof(kernel_arg_t)));
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
  cyc_comp = read_back_cycles_tag("COMP", true);

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
