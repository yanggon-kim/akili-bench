// llama2 prefill matmul-chain benchmark.
//
// Runs the same 7 matmuls/layer + 1 LM head as decoding, but with a prompt of
// length L so the matmul inner-N dimension is L (not 1).  Attention compute,
// RMSNorm, RoPE, softmax, residuals are all SKIPPED — this is a pure
// matmul-chain throughput benchmark to measure how DXA/TCU speedup scales
// with larger N.
//
// All five variants (vanilla SIMT, dense TCU, dense TCU+DXA, sparse TCU,
// sparse TCU+DXA) use the same vx_matmul(C, A, B, M, N, K) entry point.
// For the sparse variants, the first call per unique weight pointer
// internally pre-packs the 2:4 compressed data + meta into device buffers
// (see prepare_sparse_weight() in the sparse vxmath.cpp); subsequent calls
// reuse that cache.  So a single driver works for all variants.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <random>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include "vxmath.h"

struct Config {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
};

struct Weights {
    float* token_embedding;   // (vocab_size, dim)
    float* rms_att;           // (n_layers, dim)
    float* wq;                // (n_layers, dim, dim)
    float* wk;                // (n_layers, dim, kv_dim)  rows=kv_dim, cols=dim
    float* wv;                // (n_layers, dim, kv_dim)
    float* wo;                // (n_layers, dim, dim)
    float* rms_ffn;           // (n_layers, dim)
    float* w1;                // (n_layers, hidden_dim, dim)
    float* w2;                // (n_layers, dim, hidden_dim)
    float* w3;                // (n_layers, hidden_dim, dim)
    float* rms_final;         // (dim,)
    float* wcls;              // (vocab_size, dim) — may alias token_embedding
};

static void map_weights(Weights* w, const Config* p, float* ptr, int shared) {
    int head_size = p->dim / p->n_heads;
    unsigned long long nl = p->n_layers;
    w->token_embedding = ptr; ptr += (size_t)p->vocab_size * p->dim;
    w->rms_att         = ptr; ptr += nl * p->dim;
    w->wq              = ptr; ptr += nl * p->dim * (p->n_heads * head_size);
    w->wk              = ptr; ptr += nl * p->dim * (p->n_kv_heads * head_size);
    w->wv              = ptr; ptr += nl * p->dim * (p->n_kv_heads * head_size);
    w->wo              = ptr; ptr += nl * (p->n_heads * head_size) * p->dim;
    w->rms_ffn         = ptr; ptr += nl * p->dim;
    w->w1              = ptr; ptr += nl * p->dim * p->hidden_dim;
    w->w2              = ptr; ptr += nl * p->hidden_dim * p->dim;
    w->w3              = ptr; ptr += nl * p->dim * p->hidden_dim;
    w->rms_final       = ptr; ptr += p->dim;
    ptr += p->seq_len * head_size / 2;  // skip freq_cis_real
    ptr += p->seq_len * head_size / 2;  // skip freq_cis_imag
    w->wcls = shared ? w->token_embedding : ptr;
}

static float* load_checkpoint(const char* path, Config* cfg, Weights* w,
                              int* fd_out, size_t* size_out) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "open(%s) failed\n", path); exit(1); }
    struct stat st;
    if (fstat(fd, &st) != 0) { fprintf(stderr, "fstat failed\n"); exit(1); }
    size_t file_size = st.st_size;
    void* map = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) { fprintf(stderr, "mmap failed\n"); exit(1); }
    std::memcpy(cfg, map, sizeof(Config));
    int shared = cfg->vocab_size > 0 ? 1 : 0;
    cfg->vocab_size = std::abs(cfg->vocab_size);
    float* wptr = (float*)map + sizeof(Config) / sizeof(float);
    map_weights(w, cfg, wptr, shared);
    *fd_out = fd;
    *size_out = file_size;
    return (float*)map;
}

static void usage(const char* prog) {
    printf("Usage: %s [-m checkpoint] [-L prompt_len] [-v verbose] [-h]\n", prog);
    printf("  -m <path>  checkpoint file (default ../akili_llama2/data/stories15M.bin)\n");
    printf("  -L <int>   prompt length (default 128)\n");
    printf("  -v <int>   verbose level (default 0)\n");
}

int main(int argc, char** argv) {
    const char* ckpt = "../akili_llama2/data/stories15M.bin";
    int L = 128;
    int verbose = 0;

    // Simple positional or -flag argv parse (match existing benchmarks' style).
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-h")) { usage(argv[0]); return 0; }
        else if (!std::strcmp(argv[i], "-m") && i + 1 < argc) { ckpt = argv[++i]; }
        else if (!std::strcmp(argv[i], "-L") && i + 1 < argc) { L = std::atoi(argv[++i]); }
        else if (!std::strcmp(argv[i], "-v") && i + 1 < argc) { verbose = std::atoi(argv[++i]); }
        else if (argv[i][0] != '-') { ckpt = argv[i]; }  // positional
    }

    Config cfg;
    Weights w;
    int fd;
    size_t file_size;
    float* map = load_checkpoint(ckpt, &cfg, &w, &fd, &file_size);

    int dim        = cfg.dim;
    int hidden_dim = cfg.hidden_dim;
    int n_layers   = cfg.n_layers;
    int kv_dim     = (cfg.dim * cfg.n_kv_heads) / cfg.n_heads;
    int vocab_size = cfg.vocab_size;

    printf("PREFILL config: dim=%d hidden_dim=%d n_layers=%d n_heads=%d n_kv_heads=%d "
           "kv_dim=%d vocab=%d L=%d\n",
           dim, hidden_dim, n_layers, cfg.n_heads, cfg.n_kv_heads, kv_dim, vocab_size, L);

    vx_init();

    // Allocate X [dim x L] (column-major logical: K=dim, N=L) and scratch
    // Y with room for the widest output (max of dim, hidden_dim, vocab_size).
    int max_M = std::max(dim, std::max(hidden_dim, vocab_size));
    std::vector<float> X((size_t)dim * L);
    std::vector<float> Y((size_t)max_M * L);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for (auto& v : X) v = dist(rng);

    // vx_matmul(C, A, B, M, N, K): C[M×N] = A[M×K] @ B[K×N].
    // For prefill, B = X with shape [K×L], so N=L.
    // Weight layout in checkpoint (see akili_llama2/llama.cpp matmul() calls):
    //   wq[l] is [dim × dim],        wk/wv[l] are [kv_dim × dim],
    //   wo[l] is [dim × dim],        w1/w3[l] are [hidden_dim × dim],
    //   w2[l] is [dim × hidden_dim], wcls is [vocab_size × dim].

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int l = 0; l < n_layers; ++l) {
        float* wq_l = w.wq + (size_t)l * dim * dim;
        float* wk_l = w.wk + (size_t)l * dim * kv_dim;
        float* wv_l = w.wv + (size_t)l * dim * kv_dim;
        float* wo_l = w.wo + (size_t)l * dim * dim;
        float* w1_l = w.w1 + (size_t)l * dim * hidden_dim;
        float* w2_l = w.w2 + (size_t)l * dim * hidden_dim;
        float* w3_l = w.w3 + (size_t)l * dim * hidden_dim;

        // Q/K/V projections
        vx_matmul(Y.data(), wq_l, X.data(), dim,    L, dim);
        vx_matmul(Y.data(), wk_l, X.data(), kv_dim, L, dim);
        vx_matmul(Y.data(), wv_l, X.data(), kv_dim, L, dim);

        // (attention compute skipped — not part of this matmul-chain benchmark)

        // Output projection
        vx_matmul(Y.data(), wo_l, X.data(), dim,    L, dim);

        // FFN
        vx_matmul(Y.data(), w1_l, X.data(), hidden_dim, L, dim);
        vx_matmul(Y.data(), w3_l, X.data(), hidden_dim, L, dim);
        vx_matmul(Y.data(), w2_l, X.data(), dim,        L, hidden_dim);

        if (verbose > 0) {
            printf("  layer %d/%d done\n", l + 1, n_layers);
        }
    }

    // LM head
    vx_matmul(Y.data(), w.wcls, X.data(), vocab_size, L, dim);

    auto t1 = std::chrono::high_resolution_clock::now();
    long us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    printf("PREFILL[L=%d]: %ld us (%.3f ms)\n", L, us, us / 1000.0);
    printf("  matmuls=%d (7*n_layers + 1 LM head)\n", 7 * n_layers + 1);

    matmul_cleanup();

    if (map != MAP_FAILED) munmap(map, file_size);
    if (fd >= 0) close(fd);

    return 0;
}
