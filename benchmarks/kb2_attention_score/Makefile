BENCH_ROOT := $(realpath ../..)
include $(BENCH_ROOT)/config.mk

PROJECT := kb2_attention_score

SRCS    := main.cpp
VX_SRCS := kernel.cpp
OPTS    ?= -m16 -n16 -d16

include $(BENCH_ROOT)/common/common.mk
