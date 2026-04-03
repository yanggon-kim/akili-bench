#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include <math.h>
#include "common.h"

// Per-ray searchsorted: for each query value, binary search in the
// ray's sorted array to find left/right bounding indices.
// One task per ray.

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
    auto sorted  = reinterpret_cast<float*>(arg->sorted_addr);
    auto queries = reinterpret_cast<float*>(arg->query_addr);
    auto ids_left  = reinterpret_cast<uint32_t*>(arg->ids_left_addr);
    auto ids_right = reinterpret_cast<uint32_t*>(arg->ids_right_addr);

    uint32_t ray = blockIdx.x;
    uint32_t ns = arg->n_sorted;
    uint32_t nq = arg->n_queries;

    uint32_t s_offset = ray * ns;
    uint32_t q_offset = ray * nq;

    for (uint32_t qi = 0; qi < nq; ++qi) {
        float val = queries[q_offset + qi];

        // Binary search: find first index where sorted[idx] >= val
        uint32_t lo = 0, hi = ns;
        while (lo < hi) {
            uint32_t mid = (lo + hi) / 2;
            if (sorted[s_offset + mid] < val)
                lo = mid + 1;
            else
                hi = mid;
        }

        // Left index: max(0, lo - 1), right index: min(lo, ns - 1)
        uint32_t left  = (lo > 0) ? (lo - 1) : 0;
        uint32_t right = (lo < ns) ? lo : (ns - 1);
        ids_left[q_offset + qi]  = left;
        ids_right[q_offset + qi] = right;
    }
}

int main() {
    kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
    return vx_spawn_threads(1, &arg->num_tasks, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
