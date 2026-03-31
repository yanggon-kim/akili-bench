BENCH_ROOT := $(realpath ../..)
include $(BENCH_ROOT)/config.mk

PROJECT := kb2_matmul_sum_max_avgpool

SRCS    := main.cpp
VX_SRCS := kernel.cpp
OPTS    ?= -m16 -n16 -p16

include $(BENCH_ROOT)/common/common.mk
