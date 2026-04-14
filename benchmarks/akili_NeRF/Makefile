BENCH_ROOT := $(realpath ../..)
include $(BENCH_ROOT)/config.mk

PROJECT := akili_NeRF

NUM_THREADS ?= 8

# SIMT-only: no TCU flags, no softfloat linkage.
CONFIGS ?= -DNUM_THREADS=$(NUM_THREADS)

SRCS    := main.cpp
VX_SRCS := kernel.cpp

OPTS ?= -r 32 -s 8

include $(BENCH_ROOT)/common/common.mk
