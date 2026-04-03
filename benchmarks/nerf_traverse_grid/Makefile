BENCH_ROOT := $(realpath ../..)
include $(BENCH_ROOT)/config.mk

PROJECT := nerf_traverse_grid

SRCS    := main.cpp
VX_SRCS := kernel.cpp
OPTS    ?= -r16 -g8 -s16

include $(BENCH_ROOT)/common/common.mk
