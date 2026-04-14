BENCH_ROOT := $(realpath ../..)
include $(BENCH_ROOT)/config.mk

PROJECT := akili_flash_tcu_dxa

NUM_THREADS   ?= 8
NUM_TCU_LANES ?= $(NUM_THREADS)

CONFIGS ?= -DNUM_THREADS=$(NUM_THREADS) -DNUM_TCU_LANES=$(NUM_TCU_LANES) \
           -DEXT_TCU_ENABLE -DEXT_DXA_ENABLE

SRCS    := main.cpp
VX_SRCS := kernel.cpp

KERNEL_LIB := vortex2

OPTS ?= -n 16 -d 16

include $(BENCH_ROOT)/common/common.mk
