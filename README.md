# Vortex Benchmarks

Benchmark suite for the [Vortex RISC-V GPGPU](https://github.com/vortexgpgpu/vortex) processor.

Each benchmark is a self-contained directory that links against an external Vortex build — no bundled Vortex copy. Inspired by [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples).

## Benchmarks

| Benchmark | Description |
|---|---|
| `flash` | FlashAttention (TCU + SIMT implementations) |
| `attention` | Standard 3-kernel attention baseline (GEMM-Softmax-GEMM) |
| `cnn` | End-to-end CNN inference pipeline (Fashion-MNIST) |
| `acccnn` | TCU-accelerated CNN inference pipeline |
| `cnn_kernels/conv` | Convolution microbenchmark |
| `cnn_kernels/im2col` | Im2col transformation microbenchmark |
| `cnn_kernels/pool` | MaxPool microbenchmark |
| `cnn_kernels/relu` | ReLU microbenchmark |
| `cnn_kernels/sgemm` | TCU SGEMM microbenchmark (requires `-DEXT_TCU_ENABLE`) |
| `llama2` | Llama-2 transformer inference (stories15M model) |

### Single-Op Kernel Benchmarks (`kb_*`)

| Benchmark | Description |
|---|---|
| `kb_elu` | ELU activation |
| `kb_gelu` | GELU activation |
| `kb_layernorm` | Layer normalization |
| `kb_leaky_relu` | Leaky ReLU activation |
| `kb_matvec` | Matrix-vector multiplication |
| `kb_max_reduce` | Max reduction |
| `kb_mean_reduce` | Mean reduction |
| `kb_relu` | ReLU activation |
| `kb_rmsnorm` | RMS normalization |
| `kb_sgemm` | Single-precision GEMM |
| `kb_sigmoid` | Sigmoid activation |
| `kb_softmax` | Softmax |
| `kb_softplus` | Softplus activation |
| `kb_sum_reduce` | Sum reduction |
| `kb_swish` | Swish activation |
| `kb_tanh` | Tanh activation |

### Fused-Op Kernel Benchmarks (`kb2_*`)

| Benchmark | Description |
|---|---|
| `kb2_attention_score` | Fused attention score (matmul + scale + softmax) |
| `kb2_gemm_add_relu` | Fused GEMM + bias add + ReLU |
| `kb2_gemm_div_sum_scale` | Fused GEMM + division + sum + scale |
| `kb2_gemm_layernorm` | Fused GEMM + layer normalization |
| `kb2_gemm_mul_leakyrelu` | Fused GEMM + elementwise multiply + leaky ReLU |
| `kb2_gemm_rmsnorm_swiglu` | Fused GEMM + RMSNorm + SwiGLU |
| `kb2_gemm_silu_mul` | Fused GEMM + SiLU + elementwise multiply |
| `kb2_matmul_scale_residadd_clamp` | Fused matmul + scale + residual add + clamp |
| `kb2_matmul_softmax` | Fused matmul + softmax |
| `kb2_matmul_sum_max_avgpool` | Fused matmul + sum + max + average pooling |

### TCU Fused-Op Kernel Benchmarks (`kbt_*`)

Tensor core (TCU) variants of the fused-op kernels. These use Vortex's WMMA API (`vx_tensor.h`) with fp16 input / fp32 accumulate tiles, combined with SIMT post-processing for activations and normalization. Each is a direct TCU counterpart to the corresponding `kb2_*` SIMT benchmark, enabling TCU vs SIMT comparison.

Requires Vortex built with `-DEXT_TCU_ENABLE`.

| Benchmark | Operation | TCU + SIMT Phases |
|---|---|---|
| `kbt_sgemm` | C = A @ B | TCU GEMM |
| `kbt_gemm_add_relu` | ReLU(x@W + bias) | TCU GEMM + bias + ReLU |
| `kbt_gemm_mul_leakyrelu` | LeakyReLU((x@W+b)*m) | TCU GEMM + mul + LeakyReLU |
| `kbt_gemm_div_sum_scale` | scale * sum(GEMM/2) | TCU GEMM + div + sum + scale |
| `kbt_gemm_layernorm` | LayerNorm(x@W + bias) | TCU GEMM + LayerNorm |
| `kbt_attention_score` | softmax(Q@K^T / sqrt(d)) | TCU GEMM + scale + softmax |
| `kbt_matmul_softmax` | softmax(A @ B) | TCU GEMM + softmax |
| `kbt_matmul_scale_residadd_clamp` | Mish(LSE(clamp(scale(A@B)))) | TCU GEMM + scale + clamp + LSE + Mish |
| `kbt_matmul_sum_max_avgpool` | sum(A@W + bias) | TCU GEMM + bias + sum |
| `kbt_gemm_rmsnorm_swiglu` | SiLU(RMSNorm(x)@Wg) * (RMSNorm(x)@Wu) | RMSNorm + 2x TCU GEMM + SiLU*mul |
| `kbt_gemm_silu_mul` | SiLU(x@Wa+ba) * (x@Wb+bb) | 2x TCU GEMM + SiLU*mul |

**Configuring TCU size:**

```bash
# Default: NUM_TCU_LANES = NUM_THREADS = 8
make -C benchmarks/kbt_sgemm run-simx

# Change SIMT width and TCU lanes together
make -C benchmarks/kbt_sgemm NUM_THREADS=16 run-simx

# Independent: 16 SIMT threads, 4 TCU lanes (smaller tiles, more SIMT parallelism)
make -C benchmarks/kbt_sgemm NUM_THREADS=16 NUM_TCU_LANES=4 run-simx
```

> **Note:** Independent `NUM_TCU_LANES != NUM_THREADS` only works with simx. The RTL requires `NUM_TCU_LANES == NUM_THREADS` (full warp execution). A matching Vortex build is required: `CONFIGS="-DNUM_THREADS=16 -DNUM_TCU_LANES=4 -DEXT_TCU_ENABLE" make -s`.

## Prerequisites

- **Vortex** — cloned and fully built (runtime, kernel, simulator): https://github.com/vortexgpgpu/vortex
- **Vortex toolchain** — LLVM Vortex, RISC-V GNU toolchain, Verilator (installed via Vortex's `ci/toolchain_install.sh`)
- **Apptainer** — recommended for running inside the Vortex container environment

## Setup

### 1. Clone Vortex and Build the Apptainer Image

```bash
git clone --recursive https://github.com/vortexgpgpu/vortex.git
```

The Apptainer definition and build instructions are located at
[`vortex/miscs/apptainer/`](https://github.com/vortexgpgpu/vortex/tree/master/miscs/apptainer).
Refer to the
[`vortex/miscs/apptainer/README.md`](https://github.com/vortexgpgpu/vortex/blob/master/miscs/apptainer/README.md)
for full details. The short version:

```bash
cd vortex/miscs/apptainer
apptainer build --no-https vortex.sif vortex.def
```

> **Note:** On clusters with Slurm, you may need to request a compute node
> before building the image (see the Vortex Apptainer README for details).
> If the toolchain is not yet installed, follow the instructions in that README
> to run `ci/toolchain_install.sh` inside the container.

### 2. Launch the Apptainer Container

> **WARNING:** Do NOT use `run_apptainer.sh` from the Vortex repo — it contains
> hardware-specific bind mounts (Xilinx FPGA, USB devices, etc.) that will fail
> on most machines. Instead, launch the container manually with only the mounts
> you need:

```bash
apptainer shell --fakeroot --cleanenv --writable-tmpfs \
  --bind /path/to/tools:/home/tools \
  --bind /path/to/vortex:/home/vortex \
  --bind /path/to/vortex-benchmarks:/home/benchmarks \
  /path/to/vortex/miscs/apptainer/vortex.sif
```

Replace the paths above with your actual directories. For example:

```bash
apptainer shell --fakeroot --cleanenv --writable-tmpfs \
  --bind $HOME/tools:/home/tools \
  --bind $HOME/vortex:/home/vortex \
  --bind $HOME/vortex-benchmarks:/home/benchmarks \
  $HOME/vortex/miscs/apptainer/vortex.sif
```

### 3. Initialize the Toolchain (inside Apptainer)

> **IMPORTANT:** You must source the toolchain environment every time you enter
> the container. Without this step, tools like `verilator` will not be found and
> builds will fail.

```bash
cd /home/vortex/build
source ./ci/toolchain_env.sh
```

Verify the toolchain is loaded:

```bash
verilator --version    # should print Verilator 5.x
```

### 4. Build Vortex (if not already done)

```bash
cd /home/vortex
mkdir -p build && cd build
../configure --xlen=32 --tooldir=/home/tools
source ./ci/toolchain_env.sh
make -s
```

**For TCU benchmarks** (sgemm, acccnn, flash with `-t 1`), rebuild Vortex with TCU enabled:

```bash
cd /home/vortex/build
source ./ci/toolchain_env.sh
CONFIGS="-DEXT_TCU_ENABLE" make -s
```

### 5. Configure Benchmarks

```bash
cd /home/benchmarks
./configure --xlen=32 --tooldir=/home/tools --vortex=/home/vortex --build=/home/vortex/build
```

**Configure options:**

| Option | Description | Default |
|---|---|---|
| `--xlen=32\|64` | RISC-V address width | `32` |
| `--tooldir=PATH` | Toolchain install directory | `$HOME/tools` |
| `--vortex=PATH` | Vortex source root | `$VORTEX_HOME` env var |
| `--build=PATH` | Vortex build directory | `$VORTEX_HOME/build` |

### 6. Build and Run

```bash
# Build and run a single benchmark
make flash
make run-flash

# Build all benchmarks
make

# List all available benchmarks
make list
```

## Running Benchmarks

```bash
# Basic usage
make run-flash                    # FlashAttention on simx
make run-attention                # Standard attention on simx
make run-llama2                   # Llama-2 inference on simx
make run-cnn                      # CNN pipeline on simx

# Run on different simulators
make -C benchmarks/flash run-simx
make -C benchmarks/flash run-rtlsim

# Custom hardware configuration
CONFIGS="-DNUM_CORES=4 -DNUM_WARPS=2 -DNUM_THREADS=8 -DMEM_CLOCK_RATIO=4" \
    make -C benchmarks/flash run-simx OPTS="-n 64 -d 8"

# FlashAttention with Tensor Core
CONFIGS="-DNUM_CORES=4 -DNUM_WARPS=2 -DNUM_THREADS=8 -DEXT_TCU_ENABLE" \
    make -C benchmarks/flash run-simx OPTS="-n 64 -d 8 -t 1"

# Llama-2 with custom token count
make -C benchmarks/llama2 run-simx OPTS="data/stories15M.bin -z data/tokenizer.bin -n 10 -v 1"
```

## Testing

Run the test suite to verify all benchmarks build and run correctly:

```bash
# Test all benchmarks
./test.sh

# Test specific benchmarks
./test.sh flash attention llama2

# Test just CNN kernels
./test.sh cnn_kernels/conv cnn_kernels/pool cnn_kernels/relu
```

## Adding a New Benchmark

1. Create a directory under `benchmarks/`:

```bash
mkdir benchmarks/my_benchmark
```

2. Add source files and a Makefile.

**For kernel+host benchmarks** (GPU kernel cross-compiled separately):

```makefile
BENCH_ROOT := $(realpath ../..)
include $(BENCH_ROOT)/config.mk

PROJECT := my_benchmark
SRCS    := main.cpp              # host source files
VX_SRCS := kernel.cpp            # device kernel source files
OPTS    ?= -n 64                 # default runtime arguments

include $(BENCH_ROOT)/common/common.mk
```

**For pipeline benchmarks** (multi-file host binary, loads .vxbin at runtime):

```makefile
BENCH_ROOT := $(realpath ../..)
include $(BENCH_ROOT)/config.mk

PROJECT := my_pipeline
SRCS    := main.cpp pipeline.cpp layers/layer1.cpp
OPTS    ?= -n64

include $(BENCH_ROOT)/common/pipeline.mk
```

3. Build and run — the top-level Makefile auto-discovers new benchmarks:

```bash
make my_benchmark
make run-my_benchmark
make list                         # verify it appears
```

## Directory Structure

```
vortex-benchmarks/
├── configure                     # Generates config.mk
├── config.mk.in                  # Configuration template
├── Makefile                      # Top-level (auto-discovers benchmarks)
├── test.sh                       # Test suite
├── common/                       # Shared build infrastructure
│   ├── common.mk                 #   Build rules for kernel+host benchmarks
│   ├── pipeline.mk               #   Build rules for pipeline benchmarks
│   ├── helper_vortex.h           #   VX_CHECK() error macro
│   └── helper_timer.h            #   Cross-platform timing utility
├── benchmarks/                   # All benchmarks
│   ├── flash/                    #   FlashAttention
│   ├── attention/                #   Standard attention baseline
│   ├── cnn/                      #   CNN inference pipeline
│   ├── acccnn/                   #   TCU-accelerated CNN pipeline
│   ├── cnn_kernels/              #   Individual CNN kernel microbenchmarks
│   │   ├── conv/
│   │   ├── im2col/
│   │   ├── pool/
│   │   ├── relu/
│   │   └── sgemm/
│   ├── llama2/                   #   Llama-2 inference
│   ├── kb_*/                     #   16 single-op kernel benchmarks
│   ├── kb2_*/                    #   10 fused-op kernel benchmarks (SIMT)
│   └── kbt_*/                    #   11 fused-op TCU kernel benchmarks (tensor core)
```

## Contributors

- **FlashAttention / Attention**: [eyoon1131](https://github.com/eyoon1131/vortex)
- **CNN / accCNN**: [gupann](https://github.com/gupann/vortex-accelerating-cnn)
- **Llama-2**: [saursin](https://github.com/saursin/vortex/tree/llama-c)

## License

See individual benchmark files for license information.
Vortex is licensed under the [Apache 2.0 License](https://github.com/vortexgpgpu/vortex/blob/master/LICENSE).
