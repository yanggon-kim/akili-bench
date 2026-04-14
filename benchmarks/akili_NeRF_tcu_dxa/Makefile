BENCH_ROOT := $(realpath ../..)
include $(BENCH_ROOT)/config.mk

PROJECT := akili_NeRF_tcu_dxa

NUM_THREADS   ?= 8
NUM_TCU_LANES ?= $(NUM_THREADS)

# Dense TCU + DXA staged smem.
CONFIGS ?= -DNUM_THREADS=$(NUM_THREADS) -DNUM_TCU_LANES=$(NUM_TCU_LANES) \
           -DEXT_TCU_ENABLE -DEXT_DXA_ENABLE

KERNEL_LIB := vortex2

SRCS    := main.cpp
VX_SRCS := kernel.cpp

OPTS ?= -r 32 -s 8

include $(BENCH_ROOT)/common/common.mk
