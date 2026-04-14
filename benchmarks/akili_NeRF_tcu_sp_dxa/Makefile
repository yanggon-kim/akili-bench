BENCH_ROOT := $(realpath ../..)
include $(BENCH_ROOT)/config.mk

PROJECT := akili_NeRF_tcu_sp_dxa

NUM_THREADS   ?= 8
NUM_TCU_LANES ?= $(NUM_THREADS)

# Sparse TCU + DXA staged smem + meta descriptor.
CONFIGS ?= -DNUM_THREADS=$(NUM_THREADS) -DNUM_TCU_LANES=$(NUM_TCU_LANES) \
           -DEXT_TCU_ENABLE -DEXT_DXA_ENABLE

KERNEL_LIB := vortex2

SRCS    := main.cpp $(SW_COMMON_DIR)/rvfloats.cpp $(SW_COMMON_DIR)/softfloat_ext.cpp
VX_SRCS := kernel.cpp

CXXFLAGS += -I$(THIRD_PARTY_DIR)/softfloat/source/include
LDFLAGS  += $(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/softfloat.a

OPTS ?= -r 32 -s 8

include $(BENCH_ROOT)/common/common.mk
