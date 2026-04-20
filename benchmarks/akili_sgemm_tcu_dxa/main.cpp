// K-crossover sweep bench: standalone sgemm wrapper around the shared
// akili DXA matmul infrastructure (vxmath.cpp / matmul_kernel.cpp).
// Same main.cpp is used by both the dense (akili_sgemm_tcu_dxa) and sparse
// (akili_sgemm_tcu_sp_dxa) benches — the linked vxmath.cpp selects the flavor.

#include "vxmath.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <random>
#include <unistd.h>

static void usage(const char* prog) {
    std::fprintf(stderr,
        "usage: %s -m M -n N -k K [-s seed]\n"
        "  M rows of A (and C). N cols of B (and C). K inner dim.\n"
        "  Defaults: M=256 N=256 K=288 seed=1\n",
        prog);
}

int main(int argc, char** argv) {
    int M = 256, N = 256, K = 288;
    unsigned seed = 1;

    int c;
    while ((c = getopt(argc, argv, "m:n:k:s:h")) != -1) {
        switch (c) {
            case 'm': M = std::atoi(optarg); break;
            case 'n': N = std::atoi(optarg); break;
            case 'k': K = std::atoi(optarg); break;
            case 's': seed = std::atoi(optarg); break;
            case 'h': default: usage(argv[0]); return (c == 'h' ? 0 : 1);
        }
    }
    if (M <= 0 || N <= 0 || K <= 0) { usage(argv[0]); return 1; }

    std::printf("sgemm bench: M=%d N=%d K=%d seed=%u\n", M, N, K, seed);

    // Fill A (M×K) and B (K×N) with small random fp32 values — any finite range
    // works; we only care about cycle counts, not numerical stability.
    std::vector<float> A(static_cast<size_t>(M) * K);
    std::vector<float> B(static_cast<size_t>(K) * N);
    std::vector<float> C(static_cast<size_t>(M) * N, 0.0f);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (auto& v : A) v = dist(rng);
    for (auto& v : B) v = dist(rng);

    vx_init();
    vx_matmul(C.data(), A.data(), B.data(), M, N, K);
    matmul_cleanup();

    // Tiny sanity probe — print sum of |C| so we know the device wrote output.
    double abs_sum = 0.0;
    for (float v : C) abs_sum += (v < 0 ? -v : v);
    std::printf("C abs-sum = %.6g (|C| elements = %zu)\n", abs_sum, C.size());

    return 0;
}
