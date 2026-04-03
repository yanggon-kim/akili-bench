/**
 * Standalone CUDA benchmark for torch-ngp ray marching kernels.
 * Compiles and runs without Python/PyTorch — pure CUDA.
 *
 * Kernels exercised:
 *   1. kernel_near_far_from_aabb     — ray-AABB intersection
 *   2. kernel_march_rays_train       — ray marching with occupancy grid
 *   3. kernel_composite_rays_train_forward  — volume rendering compositing
 *   4. kernel_composite_rays_train_backward — gradient computation
 *
 * Build:
 *   nvcc -O3 -std=c++17 -o torch_ngp_bench torch_ngp_bench.cu
 *
 * Run:
 *   ./torch_ngp_bench
 */

#include <cuda.h>
#include <cuda_runtime.h>
// ============================================================
// Kernel 5: packbits
// ============================================================
template <typename scalar_t>
__global__ void kernel_packbits(
    const scalar_t *__restrict__ grid, const uint32_t N,
    const float density_thresh, uint8_t *bitfield
) {
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;
    const scalar_t *g = grid + n * 8;
    uint8_t bits = 0;
    #pragma unroll
    for (uint8_t i = 0; i < 8; i++)
        bits |= (g[i] > density_thresh) ? ((uint8_t)1 << i) : 0;
    bitfield[n] = bits;
}
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include "timer.cuh"

// ============================================================
// torch-ngp helper functions (copied from raymarching.cu)
// ============================================================

inline constexpr __device__ float SQRT3() { return 1.7320508075688772f; }

template <typename T>
inline __host__ __device__ T div_round_up(T val, T divisor) {
    return (val + divisor - 1) / divisor;
}

inline __host__ __device__ float signf(const float x) {
    return copysignf(1.0f, x);
}

inline __host__ __device__ float clamp(const float x, const float mn, const float mx) {
    return fminf(mx, fmaxf(mn, x));
}

inline __device__ int mip_from_pos(const float x, const float y, const float z, const float max_cascade) {
    const float mx = fmaxf(fabsf(x), fmaxf(fabsf(y), fabsf(z)));
    int exponent;
    frexpf(mx, &exponent);
    return fminf(max_cascade - 1, fmaxf(0, exponent));
}

inline __device__ int mip_from_dt(const float dt, const float H, const float max_cascade) {
    const float mx = dt * H * 0.5f;
    int exponent;
    frexpf(mx, &exponent);
    return fminf(max_cascade - 1, fmaxf(0, exponent));
}

inline __host__ __device__ uint32_t __expand_bits(uint32_t v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

inline __host__ __device__ uint32_t __morton3D(uint32_t x, uint32_t y, uint32_t z) {
    return __expand_bits(x) | (__expand_bits(y) << 1) | (__expand_bits(z) << 2);
}

// ============================================================
// Kernel 1: near_far_from_aabb
// ============================================================
template <typename scalar_t>
__global__ void kernel_near_far_from_aabb(
    const scalar_t * __restrict__ rays_o,
    const scalar_t * __restrict__ rays_d,
    const scalar_t * __restrict__ aabb,
    const uint32_t N,
    const float min_near,
    scalar_t * nears, scalar_t * fars
) {
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    const float ox = rays_o[n * 3], oy = rays_o[n * 3 + 1], oz = rays_o[n * 3 + 2];
    const float dx = rays_d[n * 3], dy = rays_d[n * 3 + 1], dz = rays_d[n * 3 + 2];
    const float rdx = 1 / dx, rdy = 1 / dy, rdz = 1 / dz;

    float near = ((dx > 0 ? aabb[0] : aabb[3]) - ox) * rdx;
    float far  = ((dx > 0 ? aabb[3] : aabb[0]) - ox) * rdx;
    near = fmaxf(near, ((dy > 0 ? aabb[1] : aabb[4]) - oy) * rdy);
    far  = fminf(far,  ((dy > 0 ? aabb[4] : aabb[1]) - oy) * rdy);
    near = fmaxf(near, ((dz > 0 ? aabb[2] : aabb[5]) - oz) * rdz);
    far  = fminf(far,  ((dz > 0 ? aabb[5] : aabb[2]) - oz) * rdz);

    nears[n] = (near < far) ? fmaxf(near, min_near) : min_near;
    fars[n]  = (near < far) ? far : min_near;
}

// ============================================================
// Kernel 2: march_rays_train
// ============================================================
template <typename scalar_t>
__global__ void kernel_march_rays_train(
    const scalar_t * __restrict__ rays_o,
    const scalar_t * __restrict__ rays_d,
    const uint8_t * __restrict__ grid,
    const float bound, const float dt_gamma, const uint32_t max_steps,
    const uint32_t N, const uint32_t C, const uint32_t H, const uint32_t M,
    const scalar_t * __restrict__ nears,
    const scalar_t * __restrict__ fars,
    scalar_t * xyzs, scalar_t * dirs, scalar_t * deltas,
    int * rays, int * counter,
    const scalar_t * __restrict__ noises
) {
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    const float ox = rays_o[n*3], oy = rays_o[n*3+1], oz = rays_o[n*3+2];
    const float dx = rays_d[n*3], dy = rays_d[n*3+1], dz = rays_d[n*3+2];
    const float rdx = 1/dx, rdy = 1/dy, rdz = 1/dz;
    const float rH = 1.0f / (float)H;
    const float H3 = H * H * H;
    const float near = nears[n], far = fars[n];
    const float noise = noises[n];
    const float dt_min = 2 * SQRT3() / max_steps;
    const float dt_max = 2 * SQRT3() * (1 << (C - 1)) / H;

    float t0 = near;
    t0 += clamp(t0 * dt_gamma, dt_min, dt_max) * noise;

    // Pass 1: count steps
    float t = t0;
    uint32_t num_steps = 0;
    while (t < far && num_steps < max_steps) {
        const float x = clamp(ox + t * dx, -bound, bound);
        const float y = clamp(oy + t * dy, -bound, bound);
        const float z = clamp(oz + t * dz, -bound, bound);
        const float dt = clamp(t * dt_gamma, dt_min, dt_max);
        const int level = max(mip_from_pos(x, y, z, C), mip_from_dt(dt, H, C));
        const float mip_bound = fminf(scalbnf(1.0f, level), bound);
        const float mip_rbound = 1 / mip_bound;
        const int nx = clamp(0.5f * (x * mip_rbound + 1) * H, 0.0f, (float)(H - 1));
        const int ny = clamp(0.5f * (y * mip_rbound + 1) * H, 0.0f, (float)(H - 1));
        const int nz = clamp(0.5f * (z * mip_rbound + 1) * H, 0.0f, (float)(H - 1));
        const uint32_t index = level * H3 + __morton3D(nx, ny, nz);
        const bool occ = grid[index / 8] & (1 << (index % 8));
        if (occ) { num_steps++; t += dt; }
        else {
            const float tx = (((nx + 0.5f + 0.5f * signf(dx)) * rH * 2 - 1) * mip_bound - x) * rdx;
            const float ty = (((ny + 0.5f + 0.5f * signf(dy)) * rH * 2 - 1) * mip_bound - y) * rdy;
            const float tz = (((nz + 0.5f + 0.5f * signf(dz)) * rH * 2 - 1) * mip_bound - z) * rdz;
            const float tt = t + fmaxf(dt_min, fminf(tx, fminf(ty, tz)));
            t = fminf(tt, far);
        }
    }

    // Allocate output slot
    uint32_t point_index = atomicAdd(counter, num_steps);
    uint32_t ray_count = atomicAdd(counter + 1, 1);
    rays[n * 3] = n;
    rays[n * 3 + 1] = point_index;
    rays[n * 3 + 2] = num_steps;

    if (point_index + num_steps > M) return;

    // Pass 2: write points
    xyzs += point_index * 3;
    dirs += point_index * 3;
    deltas += point_index * 2;
    t = t0;
    uint32_t step = 0;
    float last_t = t;
    while (t < far && step < num_steps) {
        const float x = clamp(ox + t * dx, -bound, bound);
        const float y = clamp(oy + t * dy, -bound, bound);
        const float z = clamp(oz + t * dz, -bound, bound);
        const float dt = clamp(t * dt_gamma, dt_min, dt_max);
        const int level = max(mip_from_pos(x, y, z, C), mip_from_dt(dt, H, C));
        const float mip_bound = fminf(scalbnf(1.0f, level), bound);
        const float mip_rbound = 1 / mip_bound;
        const int nx = clamp(0.5f * (x * mip_rbound + 1) * H, 0.0f, (float)(H - 1));
        const int ny = clamp(0.5f * (y * mip_rbound + 1) * H, 0.0f, (float)(H - 1));
        const int nz = clamp(0.5f * (z * mip_rbound + 1) * H, 0.0f, (float)(H - 1));
        const uint32_t index = level * H3 + __morton3D(nx, ny, nz);
        const bool occ = grid[index / 8] & (1 << (index % 8));
        if (occ) {
            xyzs[step*3] = x; xyzs[step*3+1] = y; xyzs[step*3+2] = z;
            dirs[step*3] = dx; dirs[step*3+1] = dy; dirs[step*3+2] = dz;
            deltas[step*2] = dt;
            deltas[step*2+1] = t - last_t;
            last_t = t;
            step++;
            t += dt;
        } else {
            const float tx = (((nx + 0.5f + 0.5f * signf(dx)) * rH * 2 - 1) * mip_bound - x) * rdx;
            const float ty = (((ny + 0.5f + 0.5f * signf(dy)) * rH * 2 - 1) * mip_bound - y) * rdy;
            const float tz = (((nz + 0.5f + 0.5f * signf(dz)) * rH * 2 - 1) * mip_bound - z) * rdz;
            const float tt = t + fmaxf(dt_min, fminf(tx, fminf(ty, tz)));
            t = fminf(tt, far);
        }
    }
}

// ============================================================
// Kernel 3: composite_rays_train_forward
// ============================================================
template <typename scalar_t>
__global__ void kernel_composite_rays_train_forward(
    const scalar_t * __restrict__ sigmas,
    const scalar_t * __restrict__ rgbs,
    const scalar_t * __restrict__ deltas,
    const int * __restrict__ rays,
    const uint32_t M, const uint32_t N, const float T_thresh,
    scalar_t * weights_sum, scalar_t * depth, scalar_t * image
) {
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    uint32_t index = rays[n * 3];
    uint32_t offset = rays[n * 3 + 1];
    uint32_t num_steps = rays[n * 3 + 2];

    if (num_steps == 0 || offset + num_steps > M) {
        weights_sum[index] = 0; depth[index] = 0;
        image[index*3] = image[index*3+1] = image[index*3+2] = 0;
        return;
    }

    const scalar_t *s = sigmas + offset;
    const scalar_t *c = rgbs + offset * 3;
    const scalar_t *d = deltas + offset * 2;

    scalar_t T = 1.0f;
    scalar_t r = 0, g = 0, b = 0, ws = 0, t = 0, dp = 0;

    for (uint32_t step = 0; step < num_steps; step++) {
        const scalar_t alpha = 1.0f - __expf(-s[0] * d[0]);
        const scalar_t weight = alpha * T;
        r += weight * c[0]; g += weight * c[1]; b += weight * c[2];
        t += d[1]; dp += weight * t;
        ws += weight;
        T *= 1.0f - alpha;
        if (T < T_thresh) break;
        s++; c += 3; d += 2;
    }

    weights_sum[index] = ws;
    depth[index] = dp;
    image[index*3] = r; image[index*3+1] = g; image[index*3+2] = b;
}

// ============================================================
// Kernel 4: composite_rays_train_backward
// ============================================================
template <typename scalar_t>
__global__ void kernel_composite_rays_train_backward(
    const scalar_t * __restrict__ grad_weights_sum,
    const scalar_t * __restrict__ grad_image,
    const scalar_t * __restrict__ sigmas,
    const scalar_t * __restrict__ rgbs,
    const scalar_t * __restrict__ deltas,
    const int * __restrict__ rays,
    const scalar_t * __restrict__ weights_sum,
    const scalar_t * __restrict__ image,
    const uint32_t M, const uint32_t N, const float T_thresh,
    scalar_t * grad_sigmas, scalar_t * grad_rgbs
) {
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    uint32_t index = rays[n * 3];
    uint32_t offset = rays[n * 3 + 1];
    uint32_t num_steps = rays[n * 3 + 2];
    if (num_steps == 0 || offset + num_steps > M) return;

    const scalar_t *gws = grad_weights_sum + index;
    const scalar_t *gi = grad_image + index * 3;
    const scalar_t *ws_p = weights_sum + index;
    const scalar_t *im_p = image + index * 3;
    const scalar_t *s = sigmas + offset;
    const scalar_t *c = rgbs + offset * 3;
    const scalar_t *d = deltas + offset * 2;
    scalar_t *gs = grad_sigmas + offset;
    scalar_t *gc = grad_rgbs + offset * 3;

    scalar_t T = 1.0f;
    const scalar_t r_f = im_p[0], g_f = im_p[1], b_f = im_p[2], ws_f = ws_p[0];
    scalar_t r = 0, g = 0, b = 0, ws = 0;

    for (uint32_t step = 0; step < num_steps; step++) {
        const scalar_t alpha = 1.0f - __expf(-s[0] * d[0]);
        const scalar_t weight = alpha * T;
        r += weight * c[0]; g += weight * c[1]; b += weight * c[2]; ws += weight;
        T *= 1.0f - alpha;

        gc[0] = gi[0] * weight; gc[1] = gi[1] * weight; gc[2] = gi[2] * weight;
        gs[0] = d[0] * (
            gi[0] * (T * c[0] - (r_f - r)) +
            gi[1] * (T * c[1] - (g_f - g)) +
            gi[2] * (T * c[2] - (b_f - b)) +
            gws[0] * (1 - ws_f)
        );

        if (T < T_thresh) break;
        s++; c += 3; d += 2; gs++; gc += 3;
    }
}

// No curand kernel needed — we init the grid on host

// ============================================================
// Main benchmark
// ============================================================
int main() {
    // Config (matches PyTorch benchmark)
    const uint32_t N_RAYS = 4096;
    const float BOUND = 1.0f;
    const uint32_t C = 1;        // cascades
    const uint32_t H = 128;      // grid resolution
    const uint32_t MAX_STEPS = 256;   // reduced to avoid huge output
    const float DT_GAMMA = 0.0f;     // constant step size (matches PyTorch bench)
    const float T_THRESH = 1e-4f;
    const uint32_t M = N_RAYS * MAX_STEPS;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("torch-ngp Standalone CUDA Benchmark\n");
    printf("GPU: %s\n", prop.name);
    printf("Config: %u rays, bound=%.1f, %u^3 grid, max_steps=%u, M=%u\n",
           N_RAYS, BOUND, H, MAX_STEPS, M);
    fflush(stdout);

    // Allocate
    float *d_rays_o, *d_rays_d, *d_aabb, *d_nears, *d_fars, *d_noises;
    uint8_t *d_grid;
    float *d_xyzs, *d_dirs, *d_deltas;
    int *d_rays, *d_counter;
    float *d_sigmas, *d_rgbs, *d_weights_sum, *d_depth, *d_image;
    float *d_grad_ws, *d_grad_image, *d_grad_sigmas, *d_grad_rgbs;

    CUDA_CHECK(cudaMalloc(&d_rays_o, N_RAYS * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rays_d, N_RAYS * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_aabb, 6 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_nears, N_RAYS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fars, N_RAYS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_noises, N_RAYS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grid, C * H * H * H / 8));
    CUDA_CHECK(cudaMalloc(&d_xyzs, M * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dirs, M * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_deltas, M * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rays, N_RAYS * 3 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_counter, 2 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sigmas, M * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rgbs, M * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weights_sum, N_RAYS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_depth, N_RAYS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_image, N_RAYS * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_ws, N_RAYS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_image, N_RAYS * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_sigmas, M * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_rgbs, M * 3 * sizeof(float)));

    // Init data on host, copy to device
    std::vector<float> h_rays_o(N_RAYS * 3), h_rays_d(N_RAYS * 3), h_noises(N_RAYS);
    srand(42);
    for (uint32_t i = 0; i < N_RAYS * 3; i++) {
        h_rays_o[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.2f;
        h_rays_d[i] = (float)rand() / RAND_MAX - 0.5f;
    }
    // Normalize ray directions
    for (uint32_t i = 0; i < N_RAYS; i++) {
        float *d = &h_rays_d[i * 3];
        float len = sqrtf(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
        d[0] /= len; d[1] /= len; d[2] /= len;
    }
    for (uint32_t i = 0; i < N_RAYS; i++) h_noises[i] = 0.0f;

    float h_aabb[6] = {-BOUND, -BOUND, -BOUND, BOUND, BOUND, BOUND};

    CUDA_CHECK(cudaMemcpy(d_rays_o, h_rays_o.data(), N_RAYS * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rays_d, h_rays_d.data(), N_RAYS * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_aabb, h_aabb, 6 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_noises, h_noises.data(), N_RAYS * sizeof(float), cudaMemcpyHostToDevice));

    // Init random occupancy grid on host, copy to device
    uint32_t grid_bytes = C * H * H * H / 8;
    {
        std::vector<uint8_t> h_grid(grid_bytes);
        for (uint32_t i = 0; i < grid_bytes; i++)
            h_grid[i] = (rand() % 2) ? 0xFF : 0x00;
        CUDA_CHECK(cudaMemcpy(d_grid, h_grid.data(), grid_bytes, cudaMemcpyHostToDevice));
    }

    printf("\n");

    static constexpr uint32_t N_THREAD = 128;

    // ---- Benchmark 1: near_far_from_aabb ----
    auto r1 = benchmark([&]() {
        kernel_near_far_from_aabb<float><<<div_round_up(N_RAYS, N_THREAD), N_THREAD>>>(
            d_rays_o, d_rays_d, d_aabb, N_RAYS, 0.01f, d_nears, d_fars);
    });
    printf("  kernel_near_far_from_aabb:          %8.3f ms (avg), %8.3f ms (min)\n",
           r1.mean(), r1.min());

    // Run near_far once to get valid nears/fars for march
    kernel_near_far_from_aabb<float><<<div_round_up(N_RAYS, N_THREAD), N_THREAD>>>(
        d_rays_o, d_rays_d, d_aabb, N_RAYS, 0.01f, d_nears, d_fars);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- Benchmark 2: march_rays_train ----
    auto r2 = benchmark([&]() {
        CUDA_CHECK(cudaMemset(d_counter, 0, 2 * sizeof(int)));
        kernel_march_rays_train<float><<<div_round_up(N_RAYS, N_THREAD), N_THREAD>>>(
            d_rays_o, d_rays_d, d_grid, BOUND, DT_GAMMA, MAX_STEPS,
            N_RAYS, C, H, M, d_nears, d_fars,
            d_xyzs, d_dirs, d_deltas, d_rays, d_counter, d_noises);
    });

    // Get actual point count
    int h_counter[2];
    CUDA_CHECK(cudaMemset(d_counter, 0, 2 * sizeof(int)));
    kernel_march_rays_train<float><<<div_round_up(N_RAYS, N_THREAD), N_THREAD>>>(
        d_rays_o, d_rays_d, d_grid, BOUND, DT_GAMMA, MAX_STEPS,
        N_RAYS, C, H, M, d_nears, d_fars,
        d_xyzs, d_dirs, d_deltas, d_rays, d_counter, d_noises);
    CUDA_CHECK(cudaMemcpy(h_counter, d_counter, 2 * sizeof(int), cudaMemcpyDeviceToHost));
    uint32_t total_points = h_counter[0];

    printf("  kernel_march_rays_train:            %8.3f ms (avg), %8.3f ms (min)  [%u pts]\n",
           r2.mean(), r2.min(), total_points);

    // ---- Benchmark 3: composite_rays_train_forward ----
    // Fill synthetic sigmas/rgbs
    CUDA_CHECK(cudaMemset(d_sigmas, 0, M * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_rgbs, 0, M * 3 * sizeof(float)));
    // Set some values for the valid range
    {
        std::vector<float> h_sig(total_points, 10.0f);
        std::vector<float> h_rgb(total_points * 3, 0.5f);
        CUDA_CHECK(cudaMemcpy(d_sigmas, h_sig.data(), total_points * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_rgbs, h_rgb.data(), total_points * 3 * sizeof(float), cudaMemcpyHostToDevice));
    }

    auto r3 = benchmark([&]() {
        kernel_composite_rays_train_forward<float><<<div_round_up(N_RAYS, N_THREAD), N_THREAD>>>(
            d_sigmas, d_rgbs, d_deltas, d_rays, M, N_RAYS, T_THRESH,
            d_weights_sum, d_depth, d_image);
    });
    printf("  kernel_composite_train_forward:     %8.3f ms (avg), %8.3f ms (min)\n",
           r3.mean(), r3.min());

    // ---- Benchmark 4: composite_rays_train_backward ----
    // Fill synthetic gradients
    {
        std::vector<float> h_grad_ws(N_RAYS, 1.0f);
        std::vector<float> h_grad_im(N_RAYS * 3, 1.0f);
        CUDA_CHECK(cudaMemcpy(d_grad_ws, h_grad_ws.data(), N_RAYS * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_grad_image, h_grad_im.data(), N_RAYS * 3 * sizeof(float), cudaMemcpyHostToDevice));
    }
    // Run forward once to populate weights_sum, image
    kernel_composite_rays_train_forward<float><<<div_round_up(N_RAYS, N_THREAD), N_THREAD>>>(
        d_sigmas, d_rgbs, d_deltas, d_rays, M, N_RAYS, T_THRESH,
        d_weights_sum, d_depth, d_image);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto r4 = benchmark([&]() {
        kernel_composite_rays_train_backward<float><<<div_round_up(N_RAYS, N_THREAD), N_THREAD>>>(
            d_grad_ws, d_grad_image, d_sigmas, d_rgbs, d_deltas, d_rays,
            d_weights_sum, d_image, M, N_RAYS, T_THRESH,
            d_grad_sigmas, d_grad_rgbs);
    });
    printf("  kernel_composite_train_backward:    %8.3f ms (avg), %8.3f ms (min)\n",
           r4.mean(), r4.min());

    // ---- Benchmark 5: packbits ----
    uint32_t grid_elems = C * H * H * H;
    uint32_t pack_N = grid_elems / 8;
    float *d_density_grid;
    uint8_t *d_bitfield;
    CUDA_CHECK(cudaMalloc(&d_density_grid, grid_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bitfield, pack_N));
    {
        std::vector<float> h_dgrid(grid_elems);
        for (uint32_t i = 0; i < grid_elems; i++) h_dgrid[i] = (float)rand() / RAND_MAX;
        CUDA_CHECK(cudaMemcpy(d_density_grid, h_dgrid.data(), grid_elems * sizeof(float), cudaMemcpyHostToDevice));
    }
    auto r5 = benchmark([&]() {
        kernel_packbits<float><<<div_round_up(pack_N, N_THREAD), N_THREAD>>>(
            d_density_grid, pack_N, 0.5f, d_bitfield);
    });
    printf("  kernel_packbits:                    %8.3f ms (avg), %8.3f ms (min)\n",
           r5.mean(), r5.min());
    cudaFree(d_density_grid); cudaFree(d_bitfield);

    printf("\n");
    float total_fwd = r1.mean() + r2.mean() + r3.mean();
    float total = total_fwd + r4.mean();
    printf("  Total forward pipeline:             %8.3f ms\n", total_fwd);
    printf("  Total forward + backward:           %8.3f ms\n", total);

    // Cleanup
    cudaFree(d_rays_o); cudaFree(d_rays_d); cudaFree(d_aabb);
    cudaFree(d_nears); cudaFree(d_fars); cudaFree(d_noises); cudaFree(d_grid);
    cudaFree(d_xyzs); cudaFree(d_dirs); cudaFree(d_deltas);
    cudaFree(d_rays); cudaFree(d_counter);
    cudaFree(d_sigmas); cudaFree(d_rgbs);
    cudaFree(d_weights_sum); cudaFree(d_depth); cudaFree(d_image);
    cudaFree(d_grad_ws); cudaFree(d_grad_image);
    cudaFree(d_grad_sigmas); cudaFree(d_grad_rgbs);

    return 0;
}
