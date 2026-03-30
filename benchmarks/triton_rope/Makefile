BENCH_ROOT := $(realpath ../..)
include $(BENCH_ROOT)/config.mk

PROJECT := triton_rope

SRCS    := main.cpp
VX_SRCS := kernel.cpp
OPTS    ?= -t32 -d16 -n4

include $(BENCH_ROOT)/common/common.mk
