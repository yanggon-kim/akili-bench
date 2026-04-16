BENCH_ROOT := $(realpath ../..)
include $(BENCH_ROOT)/config.mk

PROJECT := llama_tcu_sp_dxa

SRCS    := llama.cpp vxmath.cpp $(SW_COMMON_DIR)/rvfloats.cpp $(SW_COMMON_DIR)/softfloat_ext.cpp
VX_SRCS := matmul_kernel.cpp

SIMX_RT_CONFIG := $(VORTEX_BUILD)/runtime/simx_rt_config.stamp
SIMX_NUM_THREADS := $(strip $(shell [ -f "$(SIMX_RT_CONFIG)" ] && grep -o -- '-DNUM_THREADS=[0-9]*' "$(SIMX_RT_CONFIG)" | head -1 | cut -d= -f2))

NUM_THREADS ?= $(if $(SIMX_NUM_THREADS),$(SIMX_NUM_THREADS),8)
CONFIGS += -DNUM_THREADS=$(NUM_THREADS) -DEXT_TCU_ENABLE -DEXT_DXA_ENABLE -DTCU_SPARSE_ENABLE -DITYPE=fp16 -DOTYPE=fp32

KERNEL_LIB := vortex2

CXXFLAGS += -I$(THIRD_PARTY_DIR)/softfloat/source/include
LDFLAGS += $(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/softfloat.a

OPTS ?= ../akili_llama2/data/stories15M.bin -z ../akili_llama2/data/tokenizer.bin -n 40 -v 1

include $(BENCH_ROOT)/common/common.mk
