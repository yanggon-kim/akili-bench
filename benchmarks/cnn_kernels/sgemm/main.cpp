#include "common.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <rvfloats.h>
#include <string.h>
#include <tensor_cfg.h>
#include <unistd.h>
#include <util.h>
#include <vector>
#include <vortex.h>

using namespace vortex;
namespace vt = tensor;
using cfg = vt::wmma_config_t<NUM_THREADS, vt::fp16, vt::fp32>;


#define RT_CHECK(x) do { int r = (x); if (r) { \
  std::cerr << "Error: " << #x << " returned " << r << "\n"; exit(1); }} while(0)

int main() {

    vx_device_h dev;
    RT_CHECK(vx_dev_open(&dev));

    uint32_t M = 676;
    uint32_t N = 9;
    uint32_t K = 8;

    // -------- Compute padded dims (IMPORTANT) --------
    uint32_t padM = ((M + cfg::tileM - 1) / cfg::tileM) * cfg::tileM;
    uint32_t padN = ((N + cfg::tileN - 1) / cfg::tileN) * cfg::tileN;

    uint32_t effective_tileK = cfg::tileK * cfg::i_ratio;  // fp16 → tileK * 2
    uint32_t padK = ((K + effective_tileK - 1) / effective_tileK) * effective_tileK;

    std::cout << "Tile M=" << cfg::tileM
              << " N=" << cfg::tileN
              << " K_eff=" << effective_tileK << "\n";

    std::cout << "Padded dims: M=" << padM
              << " N=" << padN
              << " K=" << padK << "\n";

    std::cout << "WMMA Core Dimension: M=" << cfg::tcM << ", N=" << cfg::tcN << ", K=" << cfg::tcK << std::endl;
    std::cout << "WMMA Tile Dimension: M=" << cfg::tileM << ", N=" << cfg::tileN << ", K=" << cfg::tileK << std::endl;
    std::cout << "matrix A: " << M << "x" << K << std::endl;
    std::cout << "matrix B: " << K << "x" << N << std::endl;
    std::cout << "matrix C: " << M << "x" << N << std::endl;

    

    // -------- Prepare padded host buffers --------
    std::vector<uint16_t> hA(padM * padK, 0);
    std::vector<uint16_t> hB(padK * padN, 0);
    std::vector<float> hBias(M);

    // Fill actual region with random fp16
    for (uint32_t m = 0; m < M; m++)
        for (uint32_t k = 0; k < K; k++) {
            float f = float(rand()) / RAND_MAX;
            uint32_t u = rv_ftoh_s(*reinterpret_cast<uint32_t*>(&f), 0, nullptr);
            hA[m * padK + k] = (uint16_t)u;
        }

    for (uint32_t k = 0; k < K; k++)
        for (uint32_t n = 0; n < N; n++) {
            float f = float(rand()) / RAND_MAX;
            uint32_t u = rv_ftoh_s(*reinterpret_cast<uint32_t*>(&f), 0, nullptr);
            hB[k * padN + n] = (uint16_t)u;
        }

    for (uint32_t m = 0; m < M; m++)
        hBias[m] = 0.1f * m;

    // -------- Allocate device memory --------
    vx_buffer_h A_buf, B_buf, C_buf, bias_buf;

    RT_CHECK(vx_mem_alloc(dev, padM*padK*sizeof(uint16_t), VX_MEM_READ, &A_buf));
    RT_CHECK(vx_copy_to_dev(A_buf, hA.data(), 0, padM*padK*sizeof(uint16_t)));

    RT_CHECK(vx_mem_alloc(dev, padK*padN*sizeof(uint16_t), VX_MEM_READ, &B_buf));
    RT_CHECK(vx_copy_to_dev(B_buf, hB.data(), 0, padK*padN*sizeof(uint16_t)));

    RT_CHECK(vx_mem_alloc(dev, padM*padN*sizeof(float), VX_MEM_WRITE, &C_buf));

    RT_CHECK(vx_mem_alloc(dev, M*sizeof(float), VX_MEM_READ, &bias_buf));
    RT_CHECK(vx_copy_to_dev(bias_buf, hBias.data(), 0, M*sizeof(float)));

    // -------- Setup kernel args --------
    sgemm_arg_t args{};
    // args.M = M;
    // args.N = N;
    // args.K = K;

    args.M = padM;
    args.N = padN;
    args.K = padK;

    RT_CHECK(vx_mem_address(A_buf, &args.A_addr));
    RT_CHECK(vx_mem_address(B_buf, &args.B_addr));
    RT_CHECK(vx_mem_address(C_buf, &args.C_addr));
    RT_CHECK(vx_mem_address(bias_buf, &args.bias_addr));

    args.grid_dim[0] = padN / cfg::tileN;
    args.grid_dim[1] = padM / cfg::tileM;
    args.grid_dim[2] = 1;

    uint64_t NT;
    RT_CHECK(vx_dev_caps(dev, VX_CAPS_NUM_THREADS, &NT));
    args.block_dim[0] = NT;
    args.block_dim[1] = 1;
    args.block_dim[2] = 1;

    vx_buffer_h arg_buf;
    RT_CHECK(vx_upload_bytes(dev, &args, sizeof(args), &arg_buf));

    vx_buffer_h kernel_buf;
    RT_CHECK(vx_upload_kernel_file(dev, "kernel.vxbin", &kernel_buf));

    std::cout << "Launching kernel\n";
    // Run kernel
    RT_CHECK(vx_start(dev, kernel_buf, arg_buf));
    RT_CHECK(vx_ready_wait(dev, VX_MAX_TIMEOUT));

    // Read results
    std::vector<float> C(padM*padN);
    RT_CHECK(vx_copy_from_dev(C.data(), C_buf, 0, padM*padN*sizeof(float)));

    // cleanup
    if(dev){
        vx_mem_free(A_buf);
        vx_mem_free(B_buf);
        vx_mem_free(C_buf);
        vx_mem_free(bias_buf);
        vx_dev_close(dev);
    }
    
    std::cout << "SGEMM(fp16→fp32)+bias regression completed successfully.\n";
    return 0;
}
