BENCH_ROOT := $(realpath ../..)
include $(BENCH_ROOT)/config.mk

PROJECT := triton_crossentropy

SRCS    := main.cpp
VX_SRCS := kernel.cpp
OPTS    ?= -r64 -c32

include $(BENCH_ROOT)/common/common.mk
