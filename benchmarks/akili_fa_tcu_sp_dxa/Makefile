BENCH_ROOT := $(realpath ../..)
include $(BENCH_ROOT)/config.mk

PROJECT := akili_fa_tcu_sp_dxa

SRCS    := ../akili_fa_tcu_dxa/main.cpp $(SW_COMMON_DIR)/rvfloats.cpp $(SW_COMMON_DIR)/softfloat_ext.cpp
VX_SRCS := ../akili_fa_tcu_dxa/kernel.cpp

KERNEL_LIB := vortex2

OPTS ?= -n 512

VX_CFLAGS += -DFA_SPARSE
CXXFLAGS  += -DFA_SPARSE
CXXFLAGS  += -I$(THIRD_PARTY_DIR)/softfloat/source/include
LDFLAGS   += $(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/softfloat.a

include $(BENCH_ROOT)/common/common.mk
