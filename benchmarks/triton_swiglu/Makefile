BENCH_ROOT := $(realpath ../..)
include $(BENCH_ROOT)/config.mk

PROJECT := triton_swiglu

SRCS    := main.cpp
VX_SRCS := kernel.cpp
OPTS    ?= -r64 -c64

include $(BENCH_ROOT)/common/common.mk
