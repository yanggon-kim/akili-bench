BENCH_ROOT := $(realpath ../../..)
include $(BENCH_ROOT)/config.mk

PROJECT := cnn_sgemm

# TCU kernel requires NUM_THREADS and EXT_TCU_ENABLE
CONFIGS ?= -DNUM_THREADS=8 -DEXT_TCU_ENABLE

SRCS    := main.cpp $(SW_COMMON_DIR)/rvfloats.cpp $(SW_COMMON_DIR)/softfloat_ext.cpp
VX_SRCS := kernel_sgemm.cpp

CXXFLAGS += -I$(THIRD_PARTY_DIR)/softfloat/source/include
LDFLAGS  += $(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/softfloat.a

include $(BENCH_ROOT)/common/common.mk
