BENCH_ROOT := $(realpath ../..)
include $(BENCH_ROOT)/config.mk

PROJECT := akili_cnn

NUM_THREADS ?= 8

# SIMT-only: no TCU flags, no softfloat.
CONFIGS ?= -DNUM_THREADS=$(NUM_THREADS)

SRCS    := main.cpp
VX_SRCS := kernel.cpp

OPTS ?= -c 1 -o 8 -h 28 -w 28 -s 3

include $(BENCH_ROOT)/common/common.mk
