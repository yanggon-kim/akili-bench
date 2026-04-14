// akili_NeRF — SIMT-only full NeRF forward pass (3 stages, no TCU).
//
// One binary; three SIMT kernel launches per run:
//   Stage 1: ray setup       (AABB slab + stratified sampling)
//   Stage 2: tiny NeRF MLP   (positional encoding + 4 x 32-wide Linear/ReLU)
//   Stage 3: alpha compositing
//
// Reference: see akili_NeRF.md. nerfacc supplies the ray-sampling + compositing
// math (this file's CPU helpers match nerfacc's CUDA kernels); the MLP is a
// trimmed vanilla-NeRF (Mildenhall 2020) architecture because nerfacc itself
// does not ship a CUDA MLP — in real pipelines the MLP is torch.nn + cuBLAS.
//
// Per-stage cycles are reported via the kernel_cycles field written by the
// device main() with a vx_rdcycle() bracket, mirroring akili_attn's pattern.

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

#define RT_CHECK(_expr)                                         \
   do {                                                         \
     int _ret = _expr;                                          \
     if (0 == _ret) break;                                      \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
     cleanup();                                                 \
     exit(-1);                                                  \
   } while (false)

// ---------------------------------------------------------------------------
// Globals (buffers cleaned up in cleanup()).
// ---------------------------------------------------------------------------
const char* kernel_file = "kernel.vxbin";
uint32_t n_rays_req    = 32;
uint32_t n_samples_req = 8;

vx_device_h device = nullptr;
vx_buffer_h rays_o_buf = nullptr;
vx_buffer_h rays_d_buf = nullptr;
vx_buffer_h aabb_buf   = nullptr;
vx_buffer_h spos_buf   = nullptr;
vx_buffer_h delta_buf  = nullptr;
vx_buffer_h w0_buf = nullptr, b0_buf = nullptr;
vx_buffer_h w1_buf = nullptr, b1_buf = nullptr;
vx_buffer_h w2_buf = nullptr, b2_buf = nullptr;
vx_buffer_h w3_buf = nullptr, b3_buf = nullptr;
vx_buffer_h sigma_buf  = nullptr;
vx_buffer_h rgb_buf    = nullptr;
vx_buffer_h image_buf  = nullptr;
vx_buffer_h cycles_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
  std::cout << "akili_NeRF — SIMT full NeRF forward (ray-AABB + tiny MLP + composite)" << std::endl;
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
    if (spos_buf)   vx_mem_free(spos_buf);
    if (delta_buf)  vx_mem_free(delta_buf);
    if (w0_buf)     vx_mem_free(w0_buf);
    if (b0_buf)     vx_mem_free(b0_buf);
    if (w1_buf)     vx_mem_free(w1_buf);
    if (b1_buf)     vx_mem_free(b1_buf);
    if (w2_buf)     vx_mem_free(w2_buf);
    if (b2_buf)     vx_mem_free(b2_buf);
    if (w3_buf)     vx_mem_free(w3_buf);
    if (b3_buf)     vx_mem_free(b3_buf);
    if (sigma_buf)  vx_mem_free(sigma_buf);
    if (rgb_buf)    vx_mem_free(rgb_buf);
    if (image_buf)  vx_mem_free(image_buf);
    if (cycles_buffer) vx_mem_free(cycles_buffer);
    if (krnl_buffer) vx_mem_free(krnl_buffer);
    if (args_buffer) vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

// ---------------------------------------------------------------------------
// Compare helper
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
// CPU reference — byte-identical math to the three device kernels.
// ---------------------------------------------------------------------------
static void ray_setup_cpu(float* sample_pos, float* deltas,
                          const float* rays_o, const float* rays_d,
                          const float* aabb, uint32_t n_rays,
                          uint32_t n_samples, float min_near) {
  for (uint32_t ray = 0; ray < n_rays; ++ray) {
    float ox = rays_o[ray*3+0], oy = rays_o[ray*3+1], oz = rays_o[ray*3+2];
    float dx = rays_d[ray*3+0], dy = rays_d[ray*3+1], dz = rays_d[ray*3+2];
    float inv_dx = 1.0f / dx, inv_dy = 1.0f / dy, inv_dz = 1.0f / dz;
    float tx1 = (aabb[0] - ox) * inv_dx, tx2 = (aabb[3] - ox) * inv_dx;
    float ty1 = (aabb[1] - oy) * inv_dy, ty2 = (aabb[4] - oy) * inv_dy;
    float tz1 = (aabb[2] - oz) * inv_dz, tz2 = (aabb[5] - oz) * inv_dz;
    float tmin_x = std::min(tx1, tx2), tmax_x = std::max(tx1, tx2);
    float tmin_y = std::min(ty1, ty2), tmax_y = std::max(ty1, ty2);
    float tmin_z = std::min(tz1, tz2), tmax_z = std::max(tz1, tz2);
    float tmin = std::max(std::max(tmin_x, tmin_y), tmin_z);
    float tmax = std::min(std::min(tmax_x, tmax_y), tmax_z);
    if (tmin < min_near) tmin = min_near;
    float step = 0.0f;
    if (tmax > tmin) step = (tmax - tmin) / (float)n_samples;
    else tmax = tmin;
    for (uint32_t s = 0; s < n_samples; ++s) {
      float t = tmin + ((float)s + 0.5f) * step;
      uint32_t off = (ray * n_samples + s) * 3;
      sample_pos[off+0] = ox + t * dx;
      sample_pos[off+1] = oy + t * dy;
      sample_pos[off+2] = oz + t * dz;
      deltas[ray * n_samples + s] = step;
    }
  }
}

static void mlp_fwd_cpu(float* sigmas, float* rgbs,
                        const float* sample_pos, uint32_t n_points,
                        const float* W0, const float* B0,
                        const float* W1, const float* B1,
                        const float* W2, const float* B2,
                        const float* W3, const float* B3) {
  for (uint32_t tid = 0; tid < n_points; ++tid) {
    float x = sample_pos[tid*3+0];
    float y = sample_pos[tid*3+1];
    float z = sample_pos[tid*3+2];

    float h_a[MLP_W] = {0};
    float h_b[MLP_W] = {0};
    h_a[0] = x; h_a[1] = y; h_a[2] = z;
    float freq = 1.0f;
    for (int L = 0; L < PE_L; ++L) {
      h_a[3 + L*6 + 0] = sinf(freq * x);
      h_a[3 + L*6 + 1] = cosf(freq * x);
      h_a[3 + L*6 + 2] = sinf(freq * y);
      h_a[3 + L*6 + 3] = cosf(freq * y);
      h_a[3 + L*6 + 4] = sinf(freq * z);
      h_a[3 + L*6 + 5] = cosf(freq * z);
      freq *= 2.0f;
    }
    // Layer 0: 27 -> 32 + ReLU
    for (int j = 0; j < MLP_W; ++j) {
      float sum = B0[j];
      for (int k = 0; k < MLP_IN_DIM; ++k) sum += h_a[k] * W0[k * MLP_W + j];
      h_b[j] = (sum > 0.0f) ? sum : 0.0f;
    }
    // Layer 1: 32 -> 32 + ReLU
    for (int j = 0; j < MLP_W; ++j) {
      float sum = B1[j];
      for (int k = 0; k < MLP_W; ++k) sum += h_b[k] * W1[k * MLP_W + j];
      h_a[j] = (sum > 0.0f) ? sum : 0.0f;
    }
    // Layer 2: 32 -> 32 + ReLU
    for (int j = 0; j < MLP_W; ++j) {
      float sum = B2[j];
      for (int k = 0; k < MLP_W; ++k) sum += h_a[k] * W2[k * MLP_W + j];
      h_b[j] = (sum > 0.0f) ? sum : 0.0f;
    }
    // Layer 3: 32 -> 4 linear
    float o0 = B3[0], o1 = B3[1], o2 = B3[2], o3 = B3[3];
    for (int k = 0; k < MLP_W; ++k) {
      float hv = h_b[k];
      o0 += hv * W3[k * MLP_OUT_DIM + 0];
      o1 += hv * W3[k * MLP_OUT_DIM + 1];
      o2 += hv * W3[k * MLP_OUT_DIM + 2];
      o3 += hv * W3[k * MLP_OUT_DIM + 3];
    }
    sigmas[tid]         = logf(1.0f + expf(o0));
    rgbs[tid*3 + 0]     = 1.0f / (1.0f + expf(-o1));
    rgbs[tid*3 + 1]     = 1.0f / (1.0f + expf(-o2));
    rgbs[tid*3 + 2]     = 1.0f / (1.0f + expf(-o3));
  }
}

static void composite_cpu(float* image,
                          const float* sigmas, const float* rgbs,
                          const float* deltas, uint32_t n_rays,
                          uint32_t n_samples) {
  for (uint32_t ray = 0; ray < n_rays; ++ray) {
    float T = 1.0f;
    float R = 0, G = 0, B = 0;
    uint32_t base = ray * n_samples;
    for (uint32_t s = 0; s < n_samples; ++s) {
      float sigma = sigmas[base + s];
      float delta = deltas[base + s];
      float alpha = 1.0f - expf(-sigma * delta);
      float w = alpha * T;
      R += w * rgbs[(base+s)*3 + 0];
      G += w * rgbs[(base+s)*3 + 1];
      B += w * rgbs[(base+s)*3 + 2];
      T *= (1.0f - alpha);
    }
    image[ray*3 + 0] = R;
    image[ray*3 + 1] = G;
    image[ray*3 + 2] = B;
  }
}

// ---------------------------------------------------------------------------
static uint64_t read_back_cycles(const char* stage_tag, uint32_t num_blocks) {
  std::vector<uint32_t> h_cycles(num_blocks, 0);
  vx_copy_from_dev(h_cycles.data(), cycles_buffer, 0, num_blocks * sizeof(uint32_t));
  uint32_t max_cyc = 0;
  for (auto c : h_cycles) if (c > max_cyc) max_cyc = c;
  printf("KCYC[%s,nt=%u]: %u\n", stage_tag, (unsigned)NUM_THREADS, max_cyc);
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
  std::cout << "akili_NeRF (SIMT) — n_rays=" << n_rays
            << " n_samples=" << n_samples
            << " MLP=4x" << MLP_W << " PE_L=" << PE_L << std::endl;

  // -------------------- Host-side init --------------------
  // Rays: 2D pixel grid, origin at (0,0,-2), directions (u, v, 1) normalized,
  // offset 0.5 per pixel to guarantee no dx/dy is exactly 0.
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

  // MLP weights & biases (deterministic)
  std::srand(42);
  auto urnd = []() {
    return 0.2f * ((float)std::rand() / (float)RAND_MAX) - 0.1f;
  };
  std::vector<float> h_W0(MLP_IN_DIM * MLP_W), h_B0(MLP_W, 0.0f);
  std::vector<float> h_W1(MLP_W     * MLP_W), h_B1(MLP_W, 0.0f);
  std::vector<float> h_W2(MLP_W     * MLP_W), h_B2(MLP_W, 0.0f);
  std::vector<float> h_W3(MLP_W     * MLP_OUT_DIM), h_B3(MLP_OUT_DIM, 0.0f);
  for (auto& v : h_W0) v = urnd();
  for (auto& v : h_W1) v = urnd();
  for (auto& v : h_W2) v = urnd();
  for (auto& v : h_W3) v = urnd();
  // Nudge sigma bias so softplus output isn't degenerate.
  h_B3[0] = 0.5f;

  // -------------------- Allocate device buffers --------------------
  uint32_t rays_bytes   = n_rays * 3 * sizeof(float);
  uint32_t aabb_bytes   = 6 * sizeof(float);
  uint32_t spos_bytes   = n_points * 3 * sizeof(float);
  uint32_t delta_bytes  = n_points * sizeof(float);
  uint32_t sigma_bytes  = n_points * sizeof(float);
  uint32_t rgb_bytes    = n_points * 3 * sizeof(float);
  uint32_t image_bytes  = n_rays * 3 * sizeof(float);

  RT_CHECK(vx_mem_alloc(device, rays_bytes,  VX_MEM_READ_WRITE, &rays_o_buf));
  RT_CHECK(vx_mem_address(rays_o_buf, &kernel_arg.rays_o_addr));
  RT_CHECK(vx_mem_alloc(device, rays_bytes,  VX_MEM_READ_WRITE, &rays_d_buf));
  RT_CHECK(vx_mem_address(rays_d_buf, &kernel_arg.rays_d_addr));
  RT_CHECK(vx_mem_alloc(device, aabb_bytes,  VX_MEM_READ_WRITE, &aabb_buf));
  RT_CHECK(vx_mem_address(aabb_buf,   &kernel_arg.aabb_addr));
  RT_CHECK(vx_mem_alloc(device, spos_bytes,  VX_MEM_READ_WRITE, &spos_buf));
  RT_CHECK(vx_mem_address(spos_buf,   &kernel_arg.sample_pos_addr));
  RT_CHECK(vx_mem_alloc(device, delta_bytes, VX_MEM_READ_WRITE, &delta_buf));
  RT_CHECK(vx_mem_address(delta_buf,  &kernel_arg.deltas_addr));

  RT_CHECK(vx_mem_alloc(device, h_W0.size()*sizeof(float), VX_MEM_READ_WRITE, &w0_buf));
  RT_CHECK(vx_mem_address(w0_buf, &kernel_arg.w0_addr));
  RT_CHECK(vx_mem_alloc(device, h_B0.size()*sizeof(float), VX_MEM_READ_WRITE, &b0_buf));
  RT_CHECK(vx_mem_address(b0_buf, &kernel_arg.b0_addr));
  RT_CHECK(vx_mem_alloc(device, h_W1.size()*sizeof(float), VX_MEM_READ_WRITE, &w1_buf));
  RT_CHECK(vx_mem_address(w1_buf, &kernel_arg.w1_addr));
  RT_CHECK(vx_mem_alloc(device, h_B1.size()*sizeof(float), VX_MEM_READ_WRITE, &b1_buf));
  RT_CHECK(vx_mem_address(b1_buf, &kernel_arg.b1_addr));
  RT_CHECK(vx_mem_alloc(device, h_W2.size()*sizeof(float), VX_MEM_READ_WRITE, &w2_buf));
  RT_CHECK(vx_mem_address(w2_buf, &kernel_arg.w2_addr));
  RT_CHECK(vx_mem_alloc(device, h_B2.size()*sizeof(float), VX_MEM_READ_WRITE, &b2_buf));
  RT_CHECK(vx_mem_address(b2_buf, &kernel_arg.b2_addr));
  RT_CHECK(vx_mem_alloc(device, h_W3.size()*sizeof(float), VX_MEM_READ_WRITE, &w3_buf));
  RT_CHECK(vx_mem_address(w3_buf, &kernel_arg.w3_addr));
  RT_CHECK(vx_mem_alloc(device, h_B3.size()*sizeof(float), VX_MEM_READ_WRITE, &b3_buf));
  RT_CHECK(vx_mem_address(b3_buf, &kernel_arg.b3_addr));

  RT_CHECK(vx_mem_alloc(device, sigma_bytes, VX_MEM_READ_WRITE, &sigma_buf));
  RT_CHECK(vx_mem_address(sigma_buf, &kernel_arg.sigmas_addr));
  RT_CHECK(vx_mem_alloc(device, rgb_bytes,   VX_MEM_READ_WRITE, &rgb_buf));
  RT_CHECK(vx_mem_address(rgb_buf,   &kernel_arg.rgbs_addr));
  RT_CHECK(vx_mem_alloc(device, image_bytes, VX_MEM_READ_WRITE, &image_buf));
  RT_CHECK(vx_mem_address(image_buf, &kernel_arg.image_addr));

  // Per-stage grid/block shapes — the largest grid sets the cycles buffer size.
  // All 3 stages use block_dim = {NUM_THREADS, 1}.
  uint32_t grid_setup     = (n_rays   + NUM_THREADS - 1) / NUM_THREADS;
  uint32_t grid_mlp       = (n_points + NUM_THREADS - 1) / NUM_THREADS;
  uint32_t grid_comp      = (n_rays   + NUM_THREADS - 1) / NUM_THREADS;
  uint32_t max_blocks     = grid_setup;
  if (grid_mlp  > max_blocks) max_blocks = grid_mlp;
  if (grid_comp > max_blocks) max_blocks = grid_comp;
  RT_CHECK(vx_mem_alloc(device, max_blocks * sizeof(uint32_t),
                        VX_MEM_READ_WRITE, &cycles_buffer));
  RT_CHECK(vx_mem_address(cycles_buffer, &kernel_arg.cycles_addr));

  kernel_arg.n_rays    = n_rays;
  kernel_arg.n_samples = n_samples;
  kernel_arg.min_near  = min_near;

  // -------------------- Copy constants to device --------------------
  RT_CHECK(vx_copy_to_dev(rays_o_buf, h_rays_o.data(), 0, rays_bytes));
  RT_CHECK(vx_copy_to_dev(rays_d_buf, h_rays_d.data(), 0, rays_bytes));
  RT_CHECK(vx_copy_to_dev(aabb_buf,   h_aabb,           0, aabb_bytes));
  RT_CHECK(vx_copy_to_dev(w0_buf, h_W0.data(), 0, h_W0.size()*sizeof(float)));
  RT_CHECK(vx_copy_to_dev(b0_buf, h_B0.data(), 0, h_B0.size()*sizeof(float)));
  RT_CHECK(vx_copy_to_dev(w1_buf, h_W1.data(), 0, h_W1.size()*sizeof(float)));
  RT_CHECK(vx_copy_to_dev(b1_buf, h_B1.data(), 0, h_B1.size()*sizeof(float)));
  RT_CHECK(vx_copy_to_dev(w2_buf, h_W2.data(), 0, h_W2.size()*sizeof(float)));
  RT_CHECK(vx_copy_to_dev(b2_buf, h_B2.data(), 0, h_B2.size()*sizeof(float)));
  RT_CHECK(vx_copy_to_dev(w3_buf, h_W3.data(), 0, h_W3.size()*sizeof(float)));
  RT_CHECK(vx_copy_to_dev(b3_buf, h_B3.data(), 0, h_B3.size()*sizeof(float)));

  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
  RT_CHECK(vx_mem_alloc(device, sizeof(kernel_arg_t), VX_MEM_READ_WRITE, &args_buffer));

  int errors = 0;
  const float atol = 1e-3f;
  const float rtol = 1e-3f;

  uint64_t cyc_setup = 0, cyc_mlp = 0, cyc_comp = 0;

  uint32_t block_dim[2] = {NUM_THREADS, 1};

  // ================== Stage 1: ray setup ==================
  std::cout << "=== Stage 1: ray setup ===" << std::endl;
  kernel_arg.kernel_id = KID_RAY_SETUP;
  RT_CHECK(vx_copy_to_dev(args_buffer, &kernel_arg, 0, sizeof(kernel_arg_t)));
  {
    uint32_t grid_dim[2] = {grid_setup, 1};
    RT_CHECK(vx_start_g(device, krnl_buffer, args_buffer, 1, grid_dim, block_dim, 0));
  }
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
  cyc_setup = read_back_cycles("SETUP", grid_setup);

  std::vector<float> h_spos(n_points * 3);
  std::vector<float> h_deltas(n_points);
  RT_CHECK(vx_copy_from_dev(h_spos.data(),   spos_buf,  0, spos_bytes));
  RT_CHECK(vx_copy_from_dev(h_deltas.data(), delta_buf, 0, delta_bytes));
  {
    std::vector<float> ref_spos(n_points * 3);
    std::vector<float> ref_deltas(n_points);
    ray_setup_cpu(ref_spos.data(), ref_deltas.data(),
                  h_rays_o.data(), h_rays_d.data(), h_aabb,
                  n_rays, n_samples, min_near);
    int stage_err = 0;
    for (uint32_t i = 0; i < n_points * 3; ++i)
      if (!compare(ref_spos[i], h_spos[i], atol, rtol, (int)i, stage_err, "SPOS"))
        ++stage_err;
    for (uint32_t i = 0; i < n_points; ++i)
      if (!compare(ref_deltas[i], h_deltas[i], atol, rtol, (int)i, stage_err, "DELTA"))
        ++stage_err;
    if (stage_err) {
      std::cout << "Stage 1 found " << stage_err << " errors, continuing" << std::endl;
      errors += stage_err;
    }
  }

  // ================== Stage 2: MLP forward ==================
  std::cout << "=== Stage 2: tiny MLP forward ===" << std::endl;
  kernel_arg.kernel_id = KID_MLP_FWD;
  RT_CHECK(vx_copy_to_dev(args_buffer, &kernel_arg, 0, sizeof(kernel_arg_t)));
  {
    uint32_t grid_dim[2] = {grid_mlp, 1};
    RT_CHECK(vx_start_g(device, krnl_buffer, args_buffer, 1, grid_dim, block_dim, 0));
  }
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
  cyc_mlp = read_back_cycles("MLP", grid_mlp);

  std::vector<float> h_sigmas(n_points);
  std::vector<float> h_rgbs(n_points * 3);
  RT_CHECK(vx_copy_from_dev(h_sigmas.data(), sigma_buf, 0, sigma_bytes));
  RT_CHECK(vx_copy_from_dev(h_rgbs.data(),   rgb_buf,   0, rgb_bytes));
  {
    std::vector<float> ref_sigmas(n_points);
    std::vector<float> ref_rgbs(n_points * 3);
    mlp_fwd_cpu(ref_sigmas.data(), ref_rgbs.data(),
                h_spos.data(), n_points,
                h_W0.data(), h_B0.data(),
                h_W1.data(), h_B1.data(),
                h_W2.data(), h_B2.data(),
                h_W3.data(), h_B3.data());
    int stage_err = 0;
    for (uint32_t i = 0; i < n_points; ++i)
      if (!compare(ref_sigmas[i], h_sigmas[i], atol, rtol, (int)i, stage_err, "SIGMA"))
        ++stage_err;
    for (uint32_t i = 0; i < n_points * 3; ++i)
      if (!compare(ref_rgbs[i], h_rgbs[i], atol, rtol, (int)i, stage_err, "RGB"))
        ++stage_err;
    if (stage_err) {
      std::cout << "Stage 2 found " << stage_err << " errors, continuing" << std::endl;
      errors += stage_err;
    }
  }

  // ================== Stage 3: alpha compositing ==================
  std::cout << "=== Stage 3: alpha compositing ===" << std::endl;
  kernel_arg.kernel_id = KID_COMPOSITE;
  RT_CHECK(vx_copy_to_dev(args_buffer, &kernel_arg, 0, sizeof(kernel_arg_t)));
  {
    uint32_t grid_dim[2] = {grid_comp, 1};
    RT_CHECK(vx_start_g(device, krnl_buffer, args_buffer, 1, grid_dim, block_dim, 0));
  }
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
  cyc_comp = read_back_cycles("COMP", grid_comp);

  std::vector<float> h_image(n_rays * 3);
  RT_CHECK(vx_copy_from_dev(h_image.data(), image_buf, 0, image_bytes));
  {
    std::vector<float> ref_image(n_rays * 3);
    composite_cpu(ref_image.data(),
                  h_sigmas.data(), h_rgbs.data(), h_deltas.data(),
                  n_rays, n_samples);
    int stage_err = 0;
    for (uint32_t i = 0; i < n_rays * 3; ++i)
      if (!compare(ref_image[i], h_image[i], atol, rtol, (int)i, stage_err, "IMAGE"))
        ++stage_err;
    if (stage_err) {
      std::cout << "Stage 3 found " << stage_err << " errors" << std::endl;
      errors += stage_err;
    }
  }

  // ================== Summary ==================
  uint64_t non_gemm = cyc_setup + cyc_comp;
  uint64_t gemm     = cyc_mlp;
  uint64_t total    = non_gemm + gemm;
  printf("KCYC[SUMMARY,nt=%u]: non_gemm=%lu gemm=%lu total=%lu",
         (unsigned)NUM_THREADS,
         (unsigned long)non_gemm, (unsigned long)gemm, (unsigned long)total);
  if (total > 0) {
    double gemm_frac = (double)gemm / (double)total;
    printf("  (gemm_frac=%.3f)", gemm_frac);
  }
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
