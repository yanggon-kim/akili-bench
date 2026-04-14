BENCH_ROOT := $(realpath ../..)
include $(BENCH_ROOT)/config.mk

PROJECT := akili_flash_tcu

NUM_THREADS   ?= 8
NUM_TCU_LANES ?= $(NUM_THREADS)

# Dense TCU: ENABLE_TCU + EXT_TCU_ENABLE. No sparse, no softfloat needed.
CONFIGS ?= -DNUM_THREADS=$(NUM_THREADS) -DNUM_TCU_LANES=$(NUM_TCU_LANES) \
           -DEXT_TCU_ENABLE

SRCS    := main.cpp
VX_SRCS := kernel.cpp

KERNEL_LIB := vortex2

OPTS ?= -n 16 -d 16

include $(BENCH_ROOT)/common/common.mk
