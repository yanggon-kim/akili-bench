# CuPBoP Flash Attention benchmark
#
# This benchmark compiles flash attention CUDA code through the CuPBoP
# pipeline (CUDA → LLVM IR → Vortex RISC-V) rather than using native
# Vortex APIs. It validates that CuPBoP correctly translates:
#   - Dynamic shared memory (extern __shared__)
#   - __syncthreads()
#   - 2D grids (blockIdx.x/y, gridDim.y)
#   - expf() math
#   - Multi-tile flash attention algorithm
#
# Prerequisites:
#   Environment variables must be set (via ci/rg-ci-setup.sh inside apptainer):
#     LLVM_PREFIX, CuPBoP_PATH, VORTEX_PATH, VORTEX_HOME,
#     RISCV_TOOLCHAIN, TOOLDIR
#
# Usage:
#   make                  — build kernel + host
#   make run-simx         — build and run on Vortex simx simulator
#   make clean            — remove all generated files

KERNEL_CU := flash_attention.cu
KERNEL    := $(basename $(KERNEL_CU))
ARCH      := 64

# Default schedule: 1:1 mapping
export VORTEX_SCHEDULE_FLAG ?= 2

# Validate environment
ifndef VORTEX_PATH
  $(error VORTEX_PATH not set — run inside apptainer with ci/rg-ci-setup.sh)
endif
ifndef CuPBoP_PATH
  $(error CuPBoP_PATH not set — run inside apptainer with ci/rg-ci-setup.sh)
endif

CUDA_PATH := $(CuPBoP_PATH)/cuda-12.1

# Vortex compilation flags (64-bit RISC-V)
VX_VXFLAGS := -Xclang -target-feature -Xclang +vortex \
              -Xclang -target-feature -Xclang +zicond \
              -mllvm -disable-loop-idiom-all

VX_CFLAGS := -O3 \
  --sysroot=$(RISCV_TOOLCHAIN)/riscv64-unknown-elf \
  --gcc-toolchain=$(TOOLDIR)/riscv64-gnu-toolchain \
  -march=rv64imafd -mabi=lp64d -mcmodel=medany \
  -fno-rtti -fno-exceptions -nostartfiles -nostdlib \
  -fdata-sections -ffunction-sections \
  -I$(VORTEX_HOME)/kernel/include -I$(VORTEX_PATH)/kernel/../hw \
  -DXLEN_64 -DNDEBUG

VX_LDFLAGS := -Wl,-Bstatic,--gc-sections,-T,$(VORTEX_HOME)/kernel/scripts/link64.ld,--defsym=STARTUP_ADDR=0x080000000 \
  $(VORTEX_HOME)/build/kernel/libvortex.a \
  -L$(TOOLDIR)/libc64/lib -lm -lc \
  $(TOOLDIR)/libcrt64/lib/baremetal/libclang_rt.builtins-riscv64.a

CC  := $(LLVM_PREFIX)/bin/clang++
LLC := llc

# ---- Targets ----

.PHONY: all clean run-simx

all: host.out kernel.vxbin

# Step 1: CUDA → device + host bitcode
$(KERNEL)-cuda-nvptx64-nvidia-cuda-sm_50.bc $(KERNEL)-host-x86_64-unknown-linux-gnu.bc: $(KERNEL_CU)
	$(CC) -O0 -g -std=c++11 $< --sysroot=/ --target=x86_64-linux-gnu \
		-L$(CUDA_PATH)/lib64 --cuda-gpu-arch=sm_50 \
		-lcudart_static -ldl -lrt -pthread -save-temps -v 2>&1 || true

# Step 2: CuPBoP kernel translation
kernel.bc: $(KERNEL)-cuda-nvptx64-nvidia-cuda-sm_50.bc
	$(CuPBoP_PATH)/build/compilation/kernelTranslator $< $@

# Step 3: CuPBoP host translation
host.bc: $(KERNEL)-host-x86_64-unknown-linux-gnu.bc
	$(CuPBoP_PATH)/build/compilation/hostTranslator $< $@

host.o: host.bc
	$(LLC) --relocation-model=pic --filetype=obj $< -o $@

host_vortexrt.o: host_vortexrt.cpp
	g++ -g -O0 $< -c -o $@

# Step 4: Vortex kernel compilation
kernel.o: kernel.bc
	$(CC) $(VX_CFLAGS) $(VX_VXFLAGS) $< -c -o $@ 2>kernel.log

kernel_wrapper.o: kernel_wrapper.cpp
	$(CC) $(VX_CFLAGS) $(VX_VXFLAGS) --gcc-toolchain=$(RISCV_TOOLCHAIN) $< -c -o $@ || true

kernel.elf: kernel_wrapper.o kernel.o
	$(CC) $(VX_CFLAGS) $(VX_VXFLAGS) --gcc-toolchain=$(RISCV_TOOLCHAIN) $^ \
		$(CuPBoP_PATH)/runtime/src/vortex/kernel/cudaKernelImpl_64.o -lm $(VX_LDFLAGS) -o $@

kernel.vxbin: kernel.elf
	OBJCOPY=$(LLVM_PREFIX)/bin/llvm-objcopy $(VORTEX_HOME)/kernel/scripts/vxbin.py $< $@

kernel.dump: kernel.elf
	$(LLVM_PREFIX)/bin/llvm-objdump -D $< > $@

# Step 5: Host binary
host.out: host.o host_vortexrt.o
	g++ -g -O0 -Wall \
		-L$(CuPBoP_PATH)/build/runtime \
		-L$(CuPBoP_PATH)/build/runtime/threadPool \
		-L$(VORTEX_PATH)/runtime/ \
		-I$(VORTEX_PATH)/kernel/include \
		-o $@ -fPIC -no-pie $^ \
		-lc -lvortexRuntime -lvortex -lThreadPool -lpthread

# Step 6: Run on simx
run-simx: host.out kernel.vxbin
	PERF_CLASS=2 \
	LD_LIBRARY_PATH=$(CuPBoP_PATH)/build/runtime/threadPool:$(VORTEX_PATH)/runtime/simx:$(CuPBoP_PATH)/build/runtime:$(LD_LIBRARY_PATH) \
	./host.out

clean:
	rm -f *.out *.o *.dump *.log *.ll *.bc *.elf *.vxbin *.txt *.cui *.ii *.s *.fatbin
