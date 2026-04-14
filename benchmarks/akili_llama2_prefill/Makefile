BENCH_ROOT := $(realpath ../..)
include $(BENCH_ROOT)/config.mk

PROJECT := akili_llama2_prefill

SRCS    := main.cpp vxmath.cpp
VX_SRCS := matmul_kernel.cpp

KERNEL_LIB := vortex2

OPTS ?= ../akili_llama2/data/stories15M.bin -L 128 -v 1

include $(BENCH_ROOT)/common/common.mk
