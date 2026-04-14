BENCH_ROOT := $(realpath ../..)
include $(BENCH_ROOT)/config.mk

PROJECT := akili_attn

NUM_THREADS ?= 8

# SIMT-only: no TCU flags, no softfloat linkage.
CONFIGS ?= -DNUM_THREADS=$(NUM_THREADS)

SRCS    := main.cpp
VX_SRCS := kernel.cpp

KERNEL_LIB := vortex2

OPTS ?= -n 16 -d 16

include $(BENCH_ROOT)/common/common.mk
