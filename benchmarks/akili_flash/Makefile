BENCH_ROOT := $(realpath ../..)
include $(BENCH_ROOT)/config.mk

PROJECT := akili_flash

NUM_THREADS ?= 8

# SIMT-only: no TCU flags, no softfloat.
CONFIGS ?= -DNUM_THREADS=$(NUM_THREADS)

KERNEL_LIB := vortex2

SRCS    := main.cpp
VX_SRCS := kernel.cpp

OPTS ?= -n 16 -d 16

include $(BENCH_ROOT)/common/common.mk
