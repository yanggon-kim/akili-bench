#!/bin/bash
set -e

DEVICE=vortex
KERNEL_CU=fmha_cupbop.cu
ARCH=64

export VORTEX_SCHEDULE_FLAG=${VORTEX_SCHEDULE_FLAG:-2}

if [ -z "$VORTEX_PATH" ]; then echo "Set \$VORTEX_PATH"; exit -1; fi
if [ -z "$CuPBoP_PATH" ]; then echo "Set \$CuPBoP_PATH"; exit -1; fi

RISCV_TOOLCHAIN_FOLDER=$RISCV_TOOLCHAIN
CUDA_PATH=$CuPBoP_PATH/cuda-12.1
KERNEL=`basename $KERNEL_CU .cu`

rm -f *.out *.o *.dump *.log *.ll *.bc *.elf *.vxbin *.txt

echo "--- Generate bitcode files(.bc) for host and device"
${LLVM_PREFIX}/bin/clang++ -O0 -g -std=c++11 ./$KERNEL_CU --sysroot=/ --target=x86_64-linux-gnu -L$CUDA_PATH/lib64 --cuda-gpu-arch=sm_50 -lcudart_static -ldl -lrt -pthread -save-temps -v || true

echo "--- Generate LLVM IR"
llvm-dis $KERNEL-cuda-nvptx64-nvidia-cuda-sm_50.bc
llvm-dis $KERNEL-host-x86_64-unknown-linux-gnu.bc

echo "--- Kernel translation"
$CuPBoP_PATH/build/compilation/kernelTranslator $KERNEL-cuda-nvptx64-nvidia-cuda-sm_50.bc kernel.bc
llvm-dis kernel.bc

echo "--- Host translation"
$CuPBoP_PATH/build/compilation/hostTranslator $KERNEL-host-x86_64-unknown-linux-gnu.bc host.bc
llvm-dis host.bc
llc --relocation-model=pic --filetype=obj host.bc -o host.o

g++ -g -O0 host_vortexrt.cpp -c -o host_vortexrt.o

echo "--- Compile for $DEVICE"
if [ $DEVICE = "vortex" ]; then
    VX_VXFLAGS="-Xclang -target-feature -Xclang +vortex -Xclang -target-feature -Xclang +zicond -mllvm -disable-loop-idiom-all"
    VX_CFLAGS="-O3 --sysroot=${RISCV_TOOLCHAIN}/riscv64-unknown-elf --gcc-toolchain=${TOOLDIR}/riscv64-gnu-toolchain -march=rv64imafd -mabi=lp64d -mcmodel=medany -fno-rtti -fno-exceptions -nostartfiles -nostdlib -fdata-sections -ffunction-sections -I${VORTEX_HOME}/kernel/include -I${VORTEX_PATH}/kernel/../hw -DXLEN_64 -DNDEBUG"
    VX_LDFLAGS="-Wl,-Bstatic,--gc-sections,-T,${VORTEX_HOME}/kernel/scripts/link64.ld,--defsym=STARTUP_ADDR=0x080000000 ${VORTEX_HOME}/build/kernel/libvortex.a -L${TOOLDIR}/libc64/lib -lm -lc ${TOOLDIR}/libcrt64/lib/baremetal/libclang_rt.builtins-riscv64.a"

    echo "--- compiling kernel.bc"
    ${LLVM_PREFIX}/bin/clang++ ${VX_CFLAGS} ${VX_VXFLAGS} kernel.bc -c -o kernel.o > kernel.log 2>&1
    echo "--- compiling kernel_wrapper.cpp"
    ${LLVM_PREFIX}/bin/clang++ ${VX_CFLAGS} ${VX_VXFLAGS} --gcc-toolchain=${RISCV_TOOLCHAIN_FOLDER} ./kernel_wrapper.cpp -c -o kernel_wrapper.o || true
    echo "--- compiling kernel.elf"
    ${LLVM_PREFIX}/bin/clang++ ${VX_CFLAGS} ${VX_VXFLAGS} --gcc-toolchain=${RISCV_TOOLCHAIN_FOLDER} kernel_wrapper.o kernel.o ${CuPBoP_PATH}/runtime/src/vortex/kernel/cudaKernelImpl_64.o -lm ${VX_LDFLAGS} -o kernel.elf

    nm -C --defined-only -g kernel.elf > lookup_global_symbols.txt
    OBJCOPY=${LLVM_PREFIX}/bin/llvm-objcopy ${VORTEX_HOME}/kernel/scripts/vxbin.py kernel.elf kernel.vxbin

    echo "--- Kernel compilation completed!"

    g++ -g -O0 -Wall -L${CuPBoP_PATH}/build/runtime -L${CuPBoP_PATH}/build/runtime/threadPool -L${VORTEX_PATH}/runtime/ -I${VORTEX_PATH}/kernel/include -o host.out -fPIC -no-pie host.o host_vortexrt.o -lc -lvortexRuntime -lvortex -lThreadPool -lpthread
    echo "--- Host compilation completed!"

    export PERF_CLASS=2
    LD_LIBRARY_PATH=${CuPBoP_PATH}/build/runtime/threadPool:${VORTEX_PATH}/runtime/simx:${CuPBoP_PATH}/build/runtime:${LD_LIBRARY_PATH} ./host.out > Batch_vortex_test1.log
    echo "--- Execution completed!"
    exit -1
fi
