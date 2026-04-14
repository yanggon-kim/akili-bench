BENCH_ROOT := $(realpath ../..)
include $(BENCH_ROOT)/config.mk

PROJECT := akili_acccnn_tcu_sp_dxa

NUM_THREADS   ?= 8
NUM_TCU_LANES ?= $(NUM_THREADS)

CONFIGS ?= -DNUM_THREADS=$(NUM_THREADS) -DNUM_TCU_LANES=$(NUM_TCU_LANES) \
           -DEXT_TCU_ENABLE -DEXT_DXA_ENABLE

SRCS    := main.cpp $(SW_COMMON_DIR)/rvfloats.cpp $(SW_COMMON_DIR)/softfloat_ext.cpp
VX_SRCS := kernel.cpp

KERNEL_LIB := vortex2

CXXFLAGS += -I$(THIRD_PARTY_DIR)/softfloat/source/include
LDFLAGS  += $(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/softfloat.a

OPTS ?= -c 1 -o 8 -h 28 -w 28 -s 3

include $(BENCH_ROOT)/common/common.mk
