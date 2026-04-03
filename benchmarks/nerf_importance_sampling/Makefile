BENCH_ROOT := $(realpath ../..)
include $(BENCH_ROOT)/config.mk

PROJECT := nerf_importance_sampling

SRCS    := main.cpp
VX_SRCS := kernel.cpp
OPTS    ?= -r32 -c16 -n8

include $(BENCH_ROOT)/common/common.mk
