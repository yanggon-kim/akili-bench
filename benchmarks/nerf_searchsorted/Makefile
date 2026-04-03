BENCH_ROOT := $(realpath ../..)
include $(BENCH_ROOT)/config.mk

PROJECT := nerf_searchsorted

SRCS    := main.cpp
VX_SRCS := kernel.cpp
OPTS    ?= -r32 -s16 -q8

include $(BENCH_ROOT)/common/common.mk
