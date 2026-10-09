BENCH_ROOT := $(realpath ../..)
include $(BENCH_ROOT)/config.mk

PROJECT := llama_tcu_sp_dxa

SRCS    := llama.cpp vxmath.cpp $(SW_COMMON_DIR)/rvfloats.cpp $(SW_COMMON_DIR)/softfloat_ext.cpp
VX_SRCS := matmul_kernel.cpp

SIMX_RT_CONFIG := $(VORTEX_BUILD)/runtime/simx_rt_config.stamp
SIMX_NUM_THREADS := $(strip $(shell [ -f "$(SIMX_RT_CONFIG)" ] && grep -o -- '-DNUM_THREADS=[0-9]*' "$(SIMX_RT_CONFIG)" | head -1 | cut -d= -f2))

NUM_THREADS ?= $(if $(SIMX_NUM_THREADS),$(SIMX_NUM_THREADS),8)
CONFIGS += -DNUM_THREADS=$(NUM_THREADS) -DEXT_TCU_ENABLE -DEXT_DXA_ENABLE \
           -DTCU_SPARSE_ENABLE -DTCU_WGMMA_ENABLE -DWGMMA_NRC=32 -DWGMMA_SS -DVX_CFG_TCU_FEDP2K \
           -DITYPE=fp16 -DOTYPE=fp32

KERNEL_LIB := vortex2

CXXFLAGS += -I$(THIRD_PARTY_DIR)/softfloat/source/include
LDFLAGS += $(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/softfloat.a

# Llama-3.2-1B (converted via vortex/scripts/convert_llama3_checkpoint.py,
# "LLM3" v2 header -- see read_checkpoint() in llama.cpp). The stories15M
# legacy checkpoint still works (plain OPTS= override) since read_checkpoint
# auto-detects the header format.
OPTS ?= /home/jlaurenceleigh/models/llama-3.2-1b/llama3_1b.bin -z /home/jlaurenceleigh/models/llama-3.2-1b/llama3_1b_tokenizer.bin -n 40 -v 1

include $(BENCH_ROOT)/common/common.mk
