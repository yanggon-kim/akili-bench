BENCH_ROOT := $(realpath ../..)
include $(BENCH_ROOT)/config.mk

PROJECT := kb_matvec

SRCS    := main.cpp
VX_SRCS := kernel.cpp
OPTS    ?= -m32 -p32

include $(BENCH_ROOT)/common/common.mk
