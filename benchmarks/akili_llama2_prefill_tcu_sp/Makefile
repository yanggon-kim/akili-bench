BENCH_ROOT := $(realpath ../..)
include $(BENCH_ROOT)/config.mk

PROJECT := akili_llama2_prefill_tcu_sp

SRCS    := main.cpp vxmath.cpp $(SW_COMMON_DIR)/rvfloats.cpp $(SW_COMMON_DIR)/softfloat_ext.cpp
VX_SRCS := matmul_kernel.cpp

SIMX_RT_CONFIG := $(VORTEX_BUILD)/runtime/simx_rt_config.stamp
SIMX_NUM_THREADS := $(strip $(shell [ -f "$(SIMX_RT_CONFIG)" ] && grep -o -- '-DNUM_THREADS=[0-9]*' "$(SIMX_RT_CONFIG)" | head -1 | cut -d= -f2))

NUM_THREADS ?= 8
CONFIGS += -DNUM_THREADS=$(NUM_THREADS) -DEXT_TCU_ENABLE -DTCU_SPARSE_ENABLE -DITYPE=fp16 -DOTYPE=fp32
ifneq ($(SIMX_NUM_THREADS),)
override CONFIGS := $(filter-out -DNUM_THREADS=%,$(CONFIGS))
CONFIGS += -DNUM_THREADS=$(SIMX_NUM_THREADS)
endif

KERNEL_LIB := vortex2

CXXFLAGS += -I$(THIRD_PARTY_DIR)/softfloat/source/include
LDFLAGS += $(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/softfloat.a

OPTS ?= ../akili_llama2/data/stories15M.bin -L 128 -v 1

include $(BENCH_ROOT)/common/common.mk
