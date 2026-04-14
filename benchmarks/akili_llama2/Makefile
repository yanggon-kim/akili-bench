BENCH_ROOT := $(realpath ../..)
include $(BENCH_ROOT)/config.mk

PROJECT := akili_llama2

SIMX_RT_CONFIG := $(VORTEX_BUILD)/runtime/simx_rt_config.stamp
SIMX_NUM_THREADS := $(strip $(shell [ -f "$(SIMX_RT_CONFIG)" ] && grep -o -- '-DNUM_THREADS=[0-9]*' "$(SIMX_RT_CONFIG)" | head -1 | cut -d= -f2))

ifneq ($(SIMX_NUM_THREADS),)
CONFIGS += -DNUM_THREADS=$(SIMX_NUM_THREADS)
endif

SRCS    := llama.cpp vxmath.cpp
VX_SRCS := matmul_kernel.cpp

OPTS ?= data/stories15M.bin -z data/tokenizer.bin -n 40 -v 1

include $(BENCH_ROOT)/common/common.mk
