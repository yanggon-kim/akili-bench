BENCH_ROOT := $(realpath ../..)
include $(BENCH_ROOT)/config.mk

PROJECT := kb2_gemm_mul_leakyrelu

SRCS    := main.cpp
VX_SRCS := kernel.cpp
OPTS    ?= -m16 -n16 -p16

include $(BENCH_ROOT)/common/common.mk
