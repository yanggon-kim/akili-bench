BENCH_ROOT := $(realpath ../..)
include $(BENCH_ROOT)/config.mk

PROJECT := attention

SRCS    := main.cpp
VX_SRCS := kernel.cpp

OPTS ?= -n 64 -d 8

include $(BENCH_ROOT)/common/common.mk
