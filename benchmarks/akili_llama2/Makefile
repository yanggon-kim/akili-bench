BENCH_ROOT := $(realpath ../..)
include $(BENCH_ROOT)/config.mk

PROJECT := llama

SRCS    := llama.cpp vxmath.cpp
VX_SRCS := matmul_kernel.cpp

KERNEL_LIB := vortex2

OPTS ?= data/stories15M.bin -z data/tokenizer.bin -n 40 -v 1

include $(BENCH_ROOT)/common/common.mk
