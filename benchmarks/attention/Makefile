BENCH_ROOT := $(realpath ../..)
include $(BENCH_ROOT)/config.mk

PROJECT := attention

NUM_THREADS   ?= 8
NUM_TCU_LANES ?= $(NUM_THREADS)

# ENABLE_TCU compiles the TCU kernel paths + enables TCU in the runtime driver.
# Always defined here so the same binary can be run in SIMT or TCU mode via -t.
CONFIGS ?= -DNUM_THREADS=$(NUM_THREADS) -DNUM_TCU_LANES=$(NUM_TCU_LANES) \
           -DEXT_TCU_ENABLE -DENABLE_TCU

# Sparse TCU helpers (prune_2to4_matrix / compress_2to4_matrix) use softfloat
# via rv_htof_s — pull in rvfloats + softfloat_ext + softfloat.a.
SRCS    := main.cpp $(SW_COMMON_DIR)/rvfloats.cpp $(SW_COMMON_DIR)/softfloat_ext.cpp
VX_SRCS := kernel.cpp

CXXFLAGS += -I$(THIRD_PARTY_DIR)/softfloat/source/include
LDFLAGS  += $(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/softfloat.a

OPTS ?= -n 16 -d 16 -t 0

include $(BENCH_ROOT)/common/common.mk
