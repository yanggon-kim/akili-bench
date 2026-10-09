BENCH_ROOT := $(realpath ../..)
include $(BENCH_ROOT)/config.mk
include $(VORTEX_BUILD)/config.mk

PROJECT := e2e_conv3x3

E2E_DIR := $(BENCH_ROOT)/benchmarks/e2e_common

SRCS    := main.cpp
VX_SRCS := kernel.cpp

CXXFLAGS  += -I$(E2E_DIR) -I.
VX_CFLAGS += -I$(E2E_DIR) -I.

VX_HDRS := $(wildcard $(E2E_DIR)/*.h) common.h
HDRS    := $(VX_HDRS)

OPTS ?= -h32 -w32 -c128

KERNEL_LIB := vortex2

include $(BENCH_ROOT)/common/common_vx2.mk
