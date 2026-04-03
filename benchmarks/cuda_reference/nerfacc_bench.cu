/**
 * Standalone CUDA benchmark for nerfacc-style operations.
 * Compiles without Python/PyTorch — pure CUDA.
 *
 * Kernels exercised:
 *   1. ray_aabb_intersect_kernel  — ray-AABB intersection
 *   2. traverse_grids_kernel      — DDA ray marching through occupancy grid
 *   3. exclusive_scan_kernel      — per-ray prefix product (transmittance)
 *   4. inclusive_scan_kernel       — per-ray prefix sum (accumulation)
 *
 * Build:
 *   See Makefile target 'nerfacc_bench'
 *
 * Run:
 *   ./nerfacc_bench
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "timer.cuh"

template <typename T>
inline __host__ __device__ T div_round_up(T val, T divisor) {
    return (val + divisor - 1) / divisor;
}

// ============================================================
// Kernel 1: ray_aabb_intersect
// ============================================================
__global__ void ray_aabb_intersect_kernel(
    int32_t n_rays, float *rays_o, float *rays_d,
    float near, float far, int32_t n_grids,
    float *aabbs, float min_near,
    float *t_sorted, float *t_indices, bool *hits
) {
    int32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_rays) return;

    float ox = rays_o[tid * 3], oy = rays_o[tid * 3 + 1], oz = rays_o[tid * 3 + 2];
    float dx = rays_d[tid * 3], dy = rays_d[tid * 3 + 1], dz = rays_d[tid * 3 + 2];
    float idx = 1.0f / dx, idy = 1.0f / dy, idz = 1.0f / dz;

    for (int32_t g = 0; g < n_grids; g++) {
        float *aabb = aabbs + g * 6;
        float tmin = (aabb[idx > 0 ? 0 : 3] - ox) * idx;
        float tmax = (aabb[idx > 0 ? 3 : 0] - ox) * idx;
        float tymin = (aabb[idy > 0 ? 1 : 4] - oy) * idy;
        float tymax = (aabb[idy > 0 ? 4 : 1] - oy) * idy;
        if (tmin > tymax || tymin > tmax) { hits[tid * n_grids + g] = false; continue; }
        tmin = fmaxf(tmin, tymin); tmax = fminf(tmax, tymax);
        float tzmin = (aabb[idz > 0 ? 2 : 5] - oz) * idz;
        float tzmax = (aabb[idz > 0 ? 5 : 2] - oz) * idz;
        if (tmin > tzmax || tzmin > tmax) { hits[tid * n_grids + g] = false; continue; }
        tmin = fmaxf(tmin, fmaxf(tzmin, min_near));
        tmax = fminf(tmax, tzmax);
        hits[tid * n_grids + g] = (tmin <= tmax);
        t_sorted[tid * n_grids * 2 + g * 2] = tmin;
        t_sorted[tid * n_grids * 2 + g * 2 + 1] = tmax;
    }
}

// ============================================================
// Kernel 2: DDA grid traversal (simplified traverse_grids)
// ============================================================
__global__ void traverse_grid_kernel(
    int32_t n_rays, float *rays_o, float *rays_d,
    bool *grid, int32_t res, float *aabb,
    float step_size,
    float *nears, float *fars,
    float *t_starts, float *t_ends,
    int64_t *chunk_starts, int64_t *chunk_cnts,
    int32_t max_samples_per_ray
) {
    int32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_rays) return;

    float ox = rays_o[tid*3], oy = rays_o[tid*3+1], oz = rays_o[tid*3+2];
    float dx = rays_d[tid*3], dy = rays_d[tid*3+1], dz = rays_d[tid*3+2];
    float near = nears[tid], far = fars[tid];

    float inv_res = 1.0f / res;
    float aabb_min[3] = {aabb[0], aabb[1], aabb[2]};
    float aabb_size[3] = {aabb[3]-aabb[0], aabb[4]-aabb[1], aabb[5]-aabb[2]};

    int64_t offset = (int64_t)tid * max_samples_per_ray;
    chunk_starts[tid] = offset;

    float t = near;
    int32_t count = 0;
    while (t < far && count < max_samples_per_ray) {
        float x = ox + t * dx, y = oy + t * dy, z = oz + t * dz;
        // Grid coords
        int gx = min(res-1, max(0, (int)((x - aabb_min[0]) / aabb_size[0] * res)));
        int gy = min(res-1, max(0, (int)((y - aabb_min[1]) / aabb_size[1] * res)));
        int gz = min(res-1, max(0, (int)((z - aabb_min[2]) / aabb_size[2] * res)));
        int idx = gx * res * res + gy * res + gz;

        if (grid[idx]) {
            t_starts[offset + count] = t;
            t_ends[offset + count] = t + step_size;
            count++;
        }
        t += step_size;
    }
    chunk_cnts[tid] = count;
}

// ============================================================
// Kernel 3: Per-ray exclusive product (transmittance computation)
// ============================================================
__global__ void exclusive_prod_kernel(
    float *output, const float *input,
    const int64_t *chunk_starts, const int64_t *chunk_cnts,
    int32_t n_rays
) {
    int32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_rays) return;

    int64_t start = chunk_starts[tid];
    int64_t cnt = chunk_cnts[tid];

    float accum = 1.0f;
    for (int64_t i = 0; i < cnt; i++) {
        output[start + i] = accum;
        accum *= input[start + i];
    }
}

// ============================================================
// Kernel 4: Per-ray inclusive sum (weighted accumulation)
// ============================================================
__global__ void inclusive_sum_kernel(
    float *output, const float *input,
    const int64_t *chunk_starts, const int64_t *chunk_cnts,
    int32_t n_rays
) {
    int32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_rays) return;

    int64_t start = chunk_starts[tid];
    int64_t cnt = chunk_cnts[tid];

    float accum = 0.0f;
    for (int64_t i = 0; i < cnt; i++) {
        accum += input[start + i];
        output[start + i] = accum;
    }
}

// ============================================================
// Kernel 5: searchsorted (binary search per query into sorted arrays)
// ============================================================
__global__ void searchsorted_kernel(
    const float *sorted_vals, const int64_t *chunk_starts, const int64_t *chunk_cnts,
    const float *query_vals, const int64_t *q_chunk_starts, const int64_t *q_chunk_cnts,
    int32_t n_rays,
    int64_t *ids_left, int64_t *ids_right
) {
    int32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_rays) return;

    int64_t s_start = chunk_starts[tid], s_cnt = chunk_cnts[tid];
    int64_t q_start = q_chunk_starts[tid], q_cnt = q_chunk_cnts[tid];

    for (int64_t qi = 0; qi < q_cnt; qi++) {
        float val = query_vals[q_start + qi];
        // Binary search in sorted_vals[s_start..s_start+s_cnt)
        int64_t lo = 0, hi = s_cnt;
        while (lo < hi) {
            int64_t mid = (lo + hi) / 2;
            if (sorted_vals[s_start + mid] < val) lo = mid + 1;
            else hi = mid;
        }
        ids_left[q_start + qi] = s_start + max((int64_t)0, lo - 1);
        ids_right[q_start + qi] = s_start + min(lo, s_cnt - 1);
    }
}

// ============================================================
// Kernel 6: importance_sampling (simplified — uniform resampling within intervals)
// ============================================================
__global__ void importance_sampling_kernel(
    const float *cdfs, const int64_t *chunk_starts, const int64_t *chunk_cnts,
    int32_t n_rays, int32_t n_new_samples,
    float *new_t_starts, float *new_t_ends,
    int64_t *new_chunk_starts, int64_t *new_chunk_cnts
) {
    int32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_rays) return;

    int64_t start = chunk_starts[tid], cnt = chunk_cnts[tid];
    int64_t out_start = (int64_t)tid * n_new_samples;
    new_chunk_starts[tid] = out_start;
    new_chunk_cnts[tid] = min((int64_t)n_new_samples, cnt);

    if (cnt == 0) { new_chunk_cnts[tid] = 0; return; }

    // Uniform resampling based on CDF
    for (int32_t s = 0; s < n_new_samples && s < cnt; s++) {
        float u = ((float)s + 0.5f) / (float)n_new_samples;
        // Binary search in cdfs
        int64_t lo = 0, hi = cnt;
        while (lo < hi) {
            int64_t mid = (lo + hi) / 2;
            if (cdfs[start + mid] < u) lo = mid + 1;
            else hi = mid;
        }
        int64_t idx = min(lo, cnt - 1);
        new_t_starts[out_start + s] = (float)idx * 0.005f;
        new_t_ends[out_start + s] = (float)(idx + 1) * 0.005f;
    }
}

// ============================================================
// Main benchmark
// ============================================================
int main() {
    const int32_t N_RAYS = 4096;
    const int32_t GRID_RES = 128;
    const float STEP_SIZE = 0.005f;
    const int32_t MAX_SAMPLES = 256;
    const int32_t N_GRIDS = 1;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("nerfacc Standalone CUDA Benchmark\n");
    printf("GPU: %s\n", prop.name);
    printf("Config: %d rays, %d^3 grid, step=%.4f, max_samples=%d\n\n",
           N_RAYS, GRID_RES, STEP_SIZE, MAX_SAMPLES);

    int64_t total_samples = (int64_t)N_RAYS * MAX_SAMPLES;

    // Allocate
    float *d_rays_o, *d_rays_d, *d_aabb, *d_nears, *d_fars;
    bool *d_grid, *d_hits;
    float *d_t_sorted, *d_t_starts, *d_t_ends;
    float *d_t_indices;
    int64_t *d_chunk_starts, *d_chunk_cnts;
    float *d_scan_in, *d_scan_out;

    CUDA_CHECK(cudaMalloc(&d_rays_o, N_RAYS * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rays_d, N_RAYS * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_aabb, 6 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_nears, N_RAYS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fars, N_RAYS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grid, GRID_RES * GRID_RES * GRID_RES * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_hits, N_RAYS * N_GRIDS * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_t_sorted, N_RAYS * N_GRIDS * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_t_indices, N_RAYS * N_GRIDS * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_t_starts, total_samples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_t_ends, total_samples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_chunk_starts, N_RAYS * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_chunk_cnts, N_RAYS * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_scan_in, total_samples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scan_out, total_samples * sizeof(float)));

    // Init data
    std::vector<float> h_ro(N_RAYS * 3), h_rd(N_RAYS * 3);
    srand(42);
    for (int i = 0; i < N_RAYS * 3; i++) {
        h_ro[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.2f;
        h_rd[i] = (float)rand() / RAND_MAX - 0.5f;
    }
    for (int i = 0; i < N_RAYS; i++) {
        float *d = &h_rd[i*3];
        float l = sqrtf(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
        d[0]/=l; d[1]/=l; d[2]/=l;
    }
    float h_aabb[6] = {-1,-1,-1,1,1,1};
    int grid_total = GRID_RES * GRID_RES * GRID_RES;
    std::vector<bool> h_grid(grid_total);
    for (int i = 0; i < grid_total; i++) h_grid[i] = (rand() % 2 == 0);
    // Near/far
    std::vector<float> h_nears(N_RAYS, 0.01f), h_fars(N_RAYS, 2.0f);
    // Scan input (1-alpha values for transmittance)
    std::vector<float> h_scan_in(total_samples);
    for (int64_t i = 0; i < total_samples; i++) h_scan_in[i] = 0.9f + 0.1f * (float)rand()/RAND_MAX;

    CUDA_CHECK(cudaMemcpy(d_rays_o, h_ro.data(), N_RAYS*3*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rays_d, h_rd.data(), N_RAYS*3*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_aabb, h_aabb, 24, cudaMemcpyHostToDevice));
    // Copy bool grid (need to use uint8 intermediary)
    std::vector<uint8_t> h_grid_u8(grid_total);
    for (int i = 0; i < grid_total; i++) h_grid_u8[i] = h_grid[i] ? 1 : 0;
    CUDA_CHECK(cudaMemcpy(d_grid, h_grid_u8.data(), grid_total, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nears, h_nears.data(), N_RAYS*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_fars, h_fars.data(), N_RAYS*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scan_in, h_scan_in.data(), total_samples*4, cudaMemcpyHostToDevice));

    const int NT = 128;

    // ---- Benchmark 1: ray_aabb_intersect ----
    auto r1 = benchmark([&]() {
        ray_aabb_intersect_kernel<<<div_round_up(N_RAYS, NT), NT>>>(
            N_RAYS, d_rays_o, d_rays_d, 0.0f, 1e10f, N_GRIDS, d_aabb, 0.01f,
            d_t_sorted, d_t_indices, d_hits);
    });
    printf("  ray_aabb_intersect:    %8.3f ms (avg), %8.3f ms (min)\n", r1.mean(), r1.min());

    // ---- Benchmark 2: grid traversal ----
    auto r2 = benchmark([&]() {
        traverse_grid_kernel<<<div_round_up(N_RAYS, NT), NT>>>(
            N_RAYS, d_rays_o, d_rays_d, d_grid, GRID_RES, d_aabb,
            STEP_SIZE, d_nears, d_fars, d_t_starts, d_t_ends,
            d_chunk_starts, d_chunk_cnts, MAX_SAMPLES);
    });

    // Get actual sample count
    traverse_grid_kernel<<<div_round_up(N_RAYS, NT), NT>>>(
        N_RAYS, d_rays_o, d_rays_d, d_grid, GRID_RES, d_aabb,
        STEP_SIZE, d_nears, d_fars, d_t_starts, d_t_ends,
        d_chunk_starts, d_chunk_cnts, MAX_SAMPLES);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<int64_t> h_cnts(N_RAYS);
    CUDA_CHECK(cudaMemcpy(h_cnts.data(), d_chunk_cnts, N_RAYS*8, cudaMemcpyDeviceToHost));
    int64_t actual_samples = 0;
    for (int i = 0; i < N_RAYS; i++) actual_samples += h_cnts[i];

    printf("  traverse_grid:         %8.3f ms (avg), %8.3f ms (min)  [%ld samples]\n",
           r2.mean(), r2.min(), actual_samples);

    // ---- Benchmark 3: exclusive product (transmittance) ----
    auto r3 = benchmark([&]() {
        exclusive_prod_kernel<<<div_round_up(N_RAYS, NT), NT>>>(
            d_scan_out, d_scan_in, d_chunk_starts, d_chunk_cnts, N_RAYS);
    });
    printf("  exclusive_prod:        %8.3f ms (avg), %8.3f ms (min)  [%ld elements]\n",
           r3.mean(), r3.min(), actual_samples);

    // ---- Benchmark 4: inclusive sum (accumulation) ----
    auto r4 = benchmark([&]() {
        inclusive_sum_kernel<<<div_round_up(N_RAYS, NT), NT>>>(
            d_scan_out, d_scan_in, d_chunk_starts, d_chunk_cnts, N_RAYS);
    });
    printf("  inclusive_sum:         %8.3f ms (avg), %8.3f ms (min)  [%ld elements]\n",
           r4.mean(), r4.min(), actual_samples);

    // ---- Benchmark 5: searchsorted ----
    // Reuse chunk_starts/cnts from traverse; create query data
    int64_t total_queries = (int64_t)N_RAYS * 64;
    float *d_query_vals;
    int64_t *d_q_starts, *d_q_cnts, *d_ids_left, *d_ids_right;
    CUDA_CHECK(cudaMalloc(&d_query_vals, total_queries * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_q_starts, N_RAYS * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_q_cnts, N_RAYS * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_ids_left, total_queries * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_ids_right, total_queries * sizeof(int64_t)));
    {
        std::vector<float> h_qv(total_queries);
        std::vector<int64_t> h_qs(N_RAYS), h_qc(N_RAYS, 64);
        for (int64_t i = 0; i < total_queries; i++) h_qv[i] = (float)rand()/RAND_MAX * 2.0f;
        for (int32_t i = 0; i < N_RAYS; i++) h_qs[i] = (int64_t)i * 64;
        CUDA_CHECK(cudaMemcpy(d_query_vals, h_qv.data(), total_queries*4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_q_starts, h_qs.data(), N_RAYS*8, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_q_cnts, h_qc.data(), N_RAYS*8, cudaMemcpyHostToDevice));
    }
    auto r5 = benchmark([&]() {
        searchsorted_kernel<<<div_round_up(N_RAYS, NT), NT>>>(
            d_t_starts, d_chunk_starts, d_chunk_cnts,
            d_query_vals, d_q_starts, d_q_cnts, N_RAYS,
            d_ids_left, d_ids_right);
    });
    printf("  searchsorted:          %8.3f ms (avg), %8.3f ms (min)\n", r5.mean(), r5.min());

    // ---- Benchmark 6: importance_sampling ----
    const int32_t N_NEW_SAMPLES = 64;
    float *d_cdfs, *d_new_t_starts, *d_new_t_ends;
    int64_t *d_new_starts, *d_new_cnts;
    int64_t new_total = (int64_t)N_RAYS * N_NEW_SAMPLES;
    CUDA_CHECK(cudaMalloc(&d_cdfs, total_samples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_new_t_starts, new_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_new_t_ends, new_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_new_starts, N_RAYS * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_new_cnts, N_RAYS * sizeof(int64_t)));
    {
        // Create monotonically increasing CDF per ray
        std::vector<float> h_cdfs(total_samples);
        for (int32_t r = 0; r < N_RAYS; r++) {
            for (int32_t s = 0; s < MAX_SAMPLES; s++) {
                h_cdfs[(int64_t)r * MAX_SAMPLES + s] = (float)(s + 1) / MAX_SAMPLES;
            }
        }
        CUDA_CHECK(cudaMemcpy(d_cdfs, h_cdfs.data(), total_samples*4, cudaMemcpyHostToDevice));
    }
    auto r6 = benchmark([&]() {
        importance_sampling_kernel<<<div_round_up(N_RAYS, NT), NT>>>(
            d_cdfs, d_chunk_starts, d_chunk_cnts, N_RAYS, N_NEW_SAMPLES,
            d_new_t_starts, d_new_t_ends, d_new_starts, d_new_cnts);
    });
    printf("  importance_sampling:   %8.3f ms (avg), %8.3f ms (min)\n", r6.mean(), r6.min());

    printf("\n  Total pipeline:        %8.3f ms\n",
           r1.mean() + r2.mean() + r3.mean() + r4.mean() + r5.mean() + r6.mean());

    // Cleanup
    cudaFree(d_rays_o); cudaFree(d_rays_d); cudaFree(d_aabb);
    cudaFree(d_nears); cudaFree(d_fars); cudaFree(d_grid); cudaFree(d_hits);
    cudaFree(d_t_sorted); cudaFree(d_t_indices);
    cudaFree(d_t_starts); cudaFree(d_t_ends);
    cudaFree(d_chunk_starts); cudaFree(d_chunk_cnts);
    cudaFree(d_scan_in); cudaFree(d_scan_out);

    return 0;
}
