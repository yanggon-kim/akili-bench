# common/common.mk — Shared build rules for standard kernel+host benchmarks
#
# Each benchmark Makefile should define:
#   BENCH_ROOT  — path to benchmark repo root (usually $(realpath ../..))
#   PROJECT     — benchmark name (used as binary name)
#   SRCS        — host source files (compiled with system g++)
#   VX_SRCS     — device kernel source files (cross-compiled with LLVM Vortex)
#   OPTS        — runtime arguments (optional)
#
# Then include this file:
#   include $(BENCH_ROOT)/common/common.mk

TARGET ?= opaesim

XRT_SYN_DIR ?= $(VORTEX_HOME)/hw/syn/xilinx/xrt
XRT_DEVICE_INDEX ?= 0

VORTEX_RT_PATH ?= $(VORTEX_BUILD)/runtime
VORTEX_KN_PATH ?= $(VORTEX_BUILD)/kernel

ifeq ($(XLEN),64)
	ifeq ($(EXT_V_ENABLE),1)
		VX_CFLAGS += -march=rv64imafdv_zve64d -mabi=lp64d
	else
		VX_CFLAGS += -march=rv64imafd -mabi=lp64d
	endif
	STARTUP_ADDR ?= 0x180000000
else
	ifeq ($(EXT_V_ENABLE),1)
		VX_CFLAGS += -march=rv32imafv_zve32f -mabi=ilp32f
	else
		VX_CFLAGS += -march=rv32imaf -mabi=ilp32f
	endif
	STARTUP_ADDR ?= 0x80000000
endif

LLVM_CFLAGS += --sysroot=$(RISCV_SYSROOT)
LLVM_CFLAGS += --gcc-toolchain=$(RISCV_TOOLCHAIN_PATH)
LLVM_CFLAGS += -Xclang -target-feature -Xclang +vortex
LLVM_CFLAGS += -Xclang -target-feature -Xclang +zicond
LLVM_CFLAGS += -mllvm -disable-loop-idiom-all

VX_CC  = $(LLVM_VORTEX)/bin/clang $(LLVM_CFLAGS)
VX_CXX = $(LLVM_VORTEX)/bin/clang++ $(LLVM_CFLAGS)
VX_DP  = $(LLVM_VORTEX)/bin/llvm-objdump
VX_CP  = $(LLVM_VORTEX)/bin/llvm-objcopy

VX_CFLAGS += -O3 -mcmodel=medany -fno-rtti -fno-exceptions -nostartfiles -nostdlib -fdata-sections -ffunction-sections
VX_CFLAGS += -I$(VORTEX_HOME)/kernel/include -I$(VORTEX_BUILD)/hw -I$(SW_COMMON_DIR)
VX_CFLAGS += -DXLEN_$(XLEN)
VX_CFLAGS += -DNDEBUG
VX_CFLAGS += $(CONFIGS)

VX_LIBS += -L$(LIBC_VORTEX)/lib -lm -lc
VX_LIBS += $(LIBCRT_VORTEX)/lib/baremetal/libclang_rt.builtins-riscv$(XLEN).a

# KERNEL_LIB selects which Vortex kernel library to link against:
#   vortex  (default): classic int-main + vx_spawn_threads flow
#   vortex2: new __kernel + vx_start_g flow (supports vx_spawn2.h / __local_mem())
KERNEL_LIB ?= vortex
VX_KMU_FLAG := $(if $(filter vortex2,$(KERNEL_LIB)),-DKMU_ENABLE)
VX_STARTUP_SRC := $(VORTEX_HOME)/kernel/src/vx_start.S
KERNEL_STARTUP := $(VORTEX_HOME)/kernel/scripts/kernel_startup.sh
VX_APP_OBJS = $(addsuffix .o, $(basename $(notdir $(VX_SRCS))))

VX_LDFLAGS += -Wl,-Bstatic,--gc-sections,-T,$(VORTEX_HOME)/kernel/scripts/link$(XLEN).ld,--defsym=STARTUP_ADDR=$(STARTUP_ADDR) $(VORTEX_KN_PATH)/lib$(KERNEL_LIB).a $(VX_LIBS)

CXXFLAGS += -std=c++17 -Wall -Wextra -pedantic -Wfatal-errors
CXXFLAGS += -I$(VORTEX_HOME)/runtime/include -I$(VORTEX_BUILD)/hw -I$(SW_COMMON_DIR)
CXXFLAGS += -I$(BENCH_ROOT)/common
CXXFLAGS += $(CONFIGS)

LDFLAGS += -L$(VORTEX_RT_PATH) -lvortex

# Debugging
ifdef DEBUG
	CXXFLAGS += -g -O0
else
	CXXFLAGS += -O2 -DNDEBUG
endif

ifeq ($(TARGET), fpga)
	OPAE_DRV_PATHS ?= libopae-c.so
else
ifeq ($(TARGET), asesim)
	OPAE_DRV_PATHS ?= libopae-c-ase.so
else
ifeq ($(TARGET), opaesim)
	OPAE_DRV_PATHS ?= libopae-c-sim.so
endif
endif
endif

all: $(PROJECT) kernel.vxbin kernel.dump

kernel.dump: kernel.elf
	$(VX_DP) -D $< > $@

kernel.vxbin: kernel.elf
	OBJCOPY=$(VX_CP) $(VORTEX_HOME)/kernel/scripts/vxbin.py $< $@

ifeq ($(KERNEL_LIB),vortex2)
# vortex2 flow: compile each app source, then build startup object using
# kernel_startup.sh (which embeds the kernel_main symbol address).
%.o: %.cpp
	$(VX_CXX) $(VX_CFLAGS) -c $< -o $@

vx_start.o: $(VX_STARTUP_SRC) $(VX_APP_OBJS)
	$(VX_CXX) $(VX_CFLAGS) -DNEED_GP -DNEED_TLS -DNEED_INITFINI $(VX_KMU_FLAG) -c $(VX_STARTUP_SRC) -o $@
	$(VX_CXX) $(VX_CFLAGS) $@ $(VX_APP_OBJS) $(VX_LDFLAGS) -o $@.elf
	$(VX_CXX) $(VX_CFLAGS) $$($(KERNEL_STARTUP) $(VX_DP) $@.elf) $(VX_KMU_FLAG) -c $(VX_STARTUP_SRC) -o $@ && rm -f $@.elf

kernel.elf: vx_start.o $(VX_APP_OBJS)
	$(VX_CXX) $(VX_CFLAGS) $^ $(VX_LDFLAGS) -o kernel.elf
else
kernel.elf: $(VX_SRCS)
	$(VX_CXX) $(VX_CFLAGS) $^ $(VX_LDFLAGS) -o kernel.elf
endif

$(PROJECT): $(SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

run-simx: $(PROJECT) kernel.vxbin
	LD_LIBRARY_PATH=$(VORTEX_RT_PATH):$(LD_LIBRARY_PATH) VORTEX_DRIVER=simx ./$(PROJECT) $(OPTS)

run-rtlsim: $(PROJECT) kernel.vxbin
	LD_LIBRARY_PATH=$(VORTEX_RT_PATH):$(LD_LIBRARY_PATH) VORTEX_DRIVER=rtlsim ./$(PROJECT) $(OPTS)

run-opae: $(PROJECT) kernel.vxbin
	SCOPE_JSON_PATH=$(VORTEX_RT_PATH)/scope.json OPAE_DRV_PATHS=$(OPAE_DRV_PATHS) LD_LIBRARY_PATH=$(VORTEX_RT_PATH):$(LD_LIBRARY_PATH) VORTEX_DRIVER=opae ./$(PROJECT) $(OPTS)

run-xrt: $(PROJECT) kernel.vxbin
ifeq ($(TARGET), hw)
	SCOPE_JSON_PATH=$(FPGA_BIN_DIR)/scope.json XRT_INI_PATH=$(VORTEX_RT_PATH)/xrt/xrt.ini EMCONFIG_PATH=$(FPGA_BIN_DIR) XRT_DEVICE_INDEX=$(XRT_DEVICE_INDEX) XRT_XCLBIN_PATH=$(FPGA_BIN_DIR)/vortex_afu.xclbin LD_LIBRARY_PATH=$(XILINX_XRT)/lib:$(VORTEX_RT_PATH):$(LD_LIBRARY_PATH) VORTEX_DRIVER=xrt ./$(PROJECT) $(OPTS)
else ifeq ($(TARGET), hw_emu)
	SCOPE_JSON_PATH=$(FPGA_BIN_DIR)/scope.json XCL_EMULATION_MODE=$(TARGET) XRT_INI_PATH=$(VORTEX_RT_PATH)/xrt/xrt.ini EMCONFIG_PATH=$(FPGA_BIN_DIR) XRT_DEVICE_INDEX=$(XRT_DEVICE_INDEX) XRT_XCLBIN_PATH=$(FPGA_BIN_DIR)/vortex_afu.xclbin LD_LIBRARY_PATH=$(XILINX_XRT)/lib:$(VORTEX_RT_PATH):$(LD_LIBRARY_PATH) VORTEX_DRIVER=xrt ./$(PROJECT) $(OPTS)
else
	SCOPE_JSON_PATH=$(VORTEX_RT_PATH)/scope.json LD_LIBRARY_PATH=$(XILINX_XRT)/lib:$(VORTEX_RT_PATH):$(LD_LIBRARY_PATH) VORTEX_DRIVER=xrt ./$(PROJECT) $(OPTS)
endif

.depend: $(SRCS)
	$(CXX) $(CXXFLAGS) -MM $^ > .depend;

clean-kernel:
	rm -rf *.elf *.vxbin *.dump

clean-host:
	rm -rf $(PROJECT) *.o *.log .depend

clean: clean-kernel clean-host

ifneq ($(MAKECMDGOALS),clean)
    -include .depend
endif
