BENCH_ROOT := $(realpath ../..)
include $(BENCH_ROOT)/config.mk

PROJECT := akili_NeRF_tcu_sp

NUM_THREADS   ?= 8
NUM_TCU_LANES ?= $(NUM_THREADS)

# Sparse TCU: ENABLE_TCU + EXT_TCU_ENABLE + softfloat (prune_2to4_matrix
# uses rv_htof_s internally).
CONFIGS ?= -DNUM_THREADS=$(NUM_THREADS) -DNUM_TCU_LANES=$(NUM_TCU_LANES) \
           -DEXT_TCU_ENABLE

SRCS    := main.cpp $(SW_COMMON_DIR)/rvfloats.cpp $(SW_COMMON_DIR)/softfloat_ext.cpp
VX_SRCS := kernel.cpp

CXXFLAGS += -I$(THIRD_PARTY_DIR)/softfloat/source/include
LDFLAGS  += $(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/softfloat.a

OPTS ?= -r 32 -s 8

include $(BENCH_ROOT)/common/common.mk
