# CuPBoP Dao Flash Attention benchmark
#
# Portable CUDA implementation of the FlashAttention-2 with causal masking from
# ColfaxResearch/cutlass-kernels (dao-flash). Replaces SM90
# features (TMA, GMMA) with causal masking + online softmax.
#
# Requires CuPBoP apptainer environment (ci/rg-ci-setup.sh).

KERNEL_CU := dao_flash_cupbop.cu
KERNEL    := $(basename $(KERNEL_CU))
ARCH      := 64

export VORTEX_SCHEDULE_FLAG ?= 2

ifndef VORTEX_PATH
  $(error VORTEX_PATH not set — run inside apptainer with ci/rg-ci-setup.sh)
endif
ifndef CuPBoP_PATH
  $(error CuPBoP_PATH not set — run inside apptainer with ci/rg-ci-setup.sh)
endif

CUDA_PATH := $(CuPBoP_PATH)/cuda-12.1
VX_VXFLAGS := -Xclang -target-feature -Xclang +vortex -Xclang -target-feature -Xclang +zicond -mllvm -disable-loop-idiom-all
VX_CFLAGS := -O3 --sysroot=$(RISCV_TOOLCHAIN)/riscv64-unknown-elf --gcc-toolchain=$(TOOLDIR)/riscv64-gnu-toolchain -march=rv64imafd -mabi=lp64d -mcmodel=medany -fno-rtti -fno-exceptions -nostartfiles -nostdlib -fdata-sections -ffunction-sections -I$(VORTEX_HOME)/kernel/include -I$(VORTEX_PATH)/kernel/../hw -DXLEN_64 -DNDEBUG
VX_LDFLAGS := -Wl,-Bstatic,--gc-sections,-T,$(VORTEX_HOME)/kernel/scripts/link64.ld,--defsym=STARTUP_ADDR=0x080000000 $(VORTEX_HOME)/build/kernel/libvortex.a -L$(TOOLDIR)/libc64/lib -lm -lc $(TOOLDIR)/libcrt64/lib/baremetal/libclang_rt.builtins-riscv64.a
CC := $(LLVM_PREFIX)/bin/clang++

.PHONY: all clean run-simx

all: host.out kernel.vxbin

$(KERNEL)-cuda-nvptx64-nvidia-cuda-sm_50.bc $(KERNEL)-host-x86_64-unknown-linux-gnu.bc: $(KERNEL_CU)
	$(CC) -O0 -g -std=c++11 $< --sysroot=/ --target=x86_64-linux-gnu -L$(CUDA_PATH)/lib64 --cuda-gpu-arch=sm_50 -lcudart_static -ldl -lrt -pthread -save-temps -v 2>&1 || true

kernel.bc: $(KERNEL)-cuda-nvptx64-nvidia-cuda-sm_50.bc
	$(CuPBoP_PATH)/build/compilation/kernelTranslator $< $@

host.bc: $(KERNEL)-host-x86_64-unknown-linux-gnu.bc
	$(CuPBoP_PATH)/build/compilation/hostTranslator $< $@

host.o: host.bc
	llc --relocation-model=pic --filetype=obj $< -o $@

host_vortexrt.o: host_vortexrt.cpp
	g++ -g -O0 $< -c -o $@

kernel.o: kernel.bc
	$(CC) $(VX_CFLAGS) $(VX_VXFLAGS) $< -c -o $@ 2>kernel.log

kernel_wrapper.o: kernel_wrapper.cpp
	$(CC) $(VX_CFLAGS) $(VX_VXFLAGS) --gcc-toolchain=$(RISCV_TOOLCHAIN) $< -c -o $@ || true

kernel.elf: kernel_wrapper.o kernel.o
	$(CC) $(VX_CFLAGS) $(VX_VXFLAGS) --gcc-toolchain=$(RISCV_TOOLCHAIN) $^ $(CuPBoP_PATH)/runtime/src/vortex/kernel/cudaKernelImpl_64.o -lm $(VX_LDFLAGS) -o $@

kernel.vxbin: kernel.elf
	OBJCOPY=$(LLVM_PREFIX)/bin/llvm-objcopy $(VORTEX_HOME)/kernel/scripts/vxbin.py $< $@

host.out: host.o host_vortexrt.o
	g++ -g -O0 -Wall -L$(CuPBoP_PATH)/build/runtime -L$(CuPBoP_PATH)/build/runtime/threadPool -L$(VORTEX_PATH)/runtime/ -I$(VORTEX_PATH)/kernel/include -o $@ -fPIC -no-pie $^ -lc -lvortexRuntime -lvortex -lThreadPool -lpthread

run-simx: host.out kernel.vxbin
	PERF_CLASS=2 LD_LIBRARY_PATH=$(CuPBoP_PATH)/build/runtime/threadPool:$(VORTEX_PATH)/runtime/simx:$(CuPBoP_PATH)/build/runtime:$(LD_LIBRARY_PATH) ./host.out

clean:
	rm -f *.out *.o *.dump *.log *.ll *.bc *.elf *.vxbin *.txt *.cui *.ii *.s *.fatbin
