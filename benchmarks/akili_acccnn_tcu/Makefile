BENCH_ROOT := $(realpath ../..)
include $(BENCH_ROOT)/config.mk

PROJECT := akili_acccnn_tcu

NUM_THREADS   ?= 8
NUM_TCU_LANES ?= $(NUM_THREADS)

# Dense TCU conv: ENABLE_TCU + EXT_TCU_ENABLE. No sparse, no softfloat needed.
CONFIGS ?= -DNUM_THREADS=$(NUM_THREADS) -DNUM_TCU_LANES=$(NUM_TCU_LANES) \
           -DEXT_TCU_ENABLE

KERNEL_LIB := vortex2

SRCS    := main.cpp
VX_SRCS := kernel.cpp

KERNEL_LIB := vortex2

OPTS ?= -c 1 -o 8 -h 28 -w 28 -s 3

include $(BENCH_ROOT)/common/common.mk
