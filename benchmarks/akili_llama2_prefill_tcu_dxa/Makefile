BENCH_ROOT := $(realpath ../..)
include $(BENCH_ROOT)/config.mk

PROJECT := akili_llama2_prefill_tcu_dxa

SRCS    := main.cpp vxmath.cpp $(SW_COMMON_DIR)/rvfloats.cpp $(SW_COMMON_DIR)/softfloat_ext.cpp
VX_SRCS := matmul_kernel.cpp

NUM_THREADS ?= 8
CONFIGS += -DNUM_THREADS=$(NUM_THREADS) -DEXT_TCU_ENABLE -DEXT_DXA_ENABLE -DITYPE=fp16 -DOTYPE=fp32

KERNEL_LIB := vortex2

CXXFLAGS += -I$(THIRD_PARTY_DIR)/softfloat/source/include
LDFLAGS += $(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/softfloat.a

OPTS ?= ../akili_llama2/data/stories15M.bin -L 128 -v 1

include $(BENCH_ROOT)/common/common.mk
