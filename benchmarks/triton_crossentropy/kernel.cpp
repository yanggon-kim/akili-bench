#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include <math.h>
#include "common.h"

// Cross-Entropy Forward: loss[i] = -logits[i,label[i]] + log(sum(exp(logits[i,:])))
// Matches Liger/Triton cross_entropy_fwd_kernel.
// One task per row (batch element). Uses log-sum-exp for numerical stability.

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
    auto logits_ptr = reinterpret_cast<float*>(arg->logits_addr);
    auto labels_ptr = reinterpret_cast<int32_t*>(arg->labels_addr);
    auto loss_ptr   = reinterpret_cast<float*>(arg->loss_addr);

    uint32_t row = blockIdx.x;
    uint32_t num_classes = arg->num_classes;
    uint32_t offset = row * num_classes;
    int32_t label = labels_ptr[row];

    // 1. Find max for numerical stability
    float max_val = logits_ptr[offset];
    for (uint32_t c = 1; c < num_classes; ++c) {
        float v = logits_ptr[offset + c];
        if (v > max_val) max_val = v;
    }

    // 2. Compute log-sum-exp
    float sum_exp = 0.0f;
    for (uint32_t c = 0; c < num_classes; ++c) {
        sum_exp += expf(logits_ptr[offset + c] - max_val);
    }
    float log_sum_exp = logf(sum_exp) + max_val;

    // 3. Loss = -logits[label] + log_sum_exp
    loss_ptr[row] = -logits_ptr[offset + label] + log_sum_exp;
}

int main() {
    kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
    return vx_spawn_threads(1, &arg->num_rows, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
