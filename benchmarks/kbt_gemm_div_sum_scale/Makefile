BENCH_ROOT := $(realpath ../..)
include $(BENCH_ROOT)/config.mk

PROJECT := kbt_gemm_div_sum_scale
NUM_THREADS ?= 8
NUM_TCU_LANES ?= $(NUM_THREADS)
CONFIGS ?= -DNUM_THREADS=$(NUM_THREADS) -DNUM_TCU_LANES=$(NUM_TCU_LANES) -DEXT_TCU_ENABLE

SRCS    := main.cpp
VX_SRCS := kernel.cpp
OPTS    ?= -m16 -n16 -p16

include $(BENCH_ROOT)/common/common.mk
