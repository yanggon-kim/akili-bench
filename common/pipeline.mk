# common/pipeline.mk — Build rules for pipeline-style benchmarks (CNN, accCNN)
#
# Pipeline benchmarks differ from standard benchmarks:
#   - Multi-file host binary (no separate kernel.elf cross-compilation step)
#   - GPU kernels are pre-built .vxbin files loaded dynamically at runtime
#   - May need data directory symlinks (images/, weights/)
#
# Each benchmark Makefile should define:
#   BENCH_ROOT  — path to benchmark repo root
#   PROJECT     — benchmark name
#   SRCS        — all host source files (main.cpp, pipeline.cpp, layers/*.cpp, etc.)
#   OPTS        — runtime arguments (optional)
#   DATA_DIR    — directory containing data files (optional, for symlinking)
#
# Then include this file:
#   include $(BENCH_ROOT)/common/pipeline.mk

TARGET ?= opaesim

XRT_SYN_DIR ?= $(VORTEX_HOME)/hw/syn/xilinx/xrt
XRT_DEVICE_INDEX ?= 0

VORTEX_RT_PATH ?= $(VORTEX_BUILD)/runtime
VORTEX_KN_PATH ?= $(VORTEX_BUILD)/kernel

ifeq ($(XLEN),64)
	ifeq ($(EXT_V_ENABLE),1)
		VX_CFLAGS += -march=rv64imafdv_zve64d -mabi=lp64d
	else
		VX_CFLAGS += -march=rv64imafd -mabi=lp64d
	endif
else
	ifeq ($(EXT_V_ENABLE),1)
		VX_CFLAGS += -march=rv32imafv_zve32f -mabi=ilp32f
	else
		VX_CFLAGS += -march=rv32imaf -mabi=ilp32f
	endif
endif

CXXFLAGS += -std=c++17 -Wall -Wextra -pedantic -Wfatal-errors
CXXFLAGS += -I$(VORTEX_HOME)/runtime/include -I$(VORTEX_BUILD)/hw -I$(SW_COMMON_DIR)
CXXFLAGS += -I$(BENCH_ROOT)/common
CXXFLAGS += $(CONFIGS)

LDFLAGS += -L$(VORTEX_RT_PATH) -lvortex

# Debugging
ifdef DEBUG
	CXXFLAGS += -g -O0
else
	CXXFLAGS += -O2 -DNDEBUG
endif

all: $(PROJECT)

$(PROJECT): $(SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

.PHONY: run-simx run-rtlsim run-opae run-xrt prepare-data clean

prepare-data:
ifdef DATA_DIR
	@if [ ! -e images ] && [ -d $(DATA_DIR)/images ]; then ln -s $(DATA_DIR)/images .; fi
	@if [ ! -e weights ] && [ -d $(DATA_DIR)/weights ]; then ln -s $(DATA_DIR)/weights .; fi
	@if [ ! -e layers ] && [ -d $(DATA_DIR)/layers ]; then ln -s $(DATA_DIR)/layers .; fi
endif

run-simx: prepare-data $(PROJECT)
	LD_LIBRARY_PATH=$(VORTEX_RT_PATH):$(LD_LIBRARY_PATH) VORTEX_DRIVER=simx ./$(PROJECT) $(OPTS)

run-rtlsim: prepare-data $(PROJECT)
	LD_LIBRARY_PATH=$(VORTEX_RT_PATH):$(LD_LIBRARY_PATH) VORTEX_DRIVER=rtlsim ./$(PROJECT) $(OPTS)

run-opae: prepare-data $(PROJECT)
	OPAE_DRV_PATHS=$(OPAE_DRV_PATHS) LD_LIBRARY_PATH=$(VORTEX_RT_PATH):$(LD_LIBRARY_PATH) VORTEX_DRIVER=opae ./$(PROJECT) $(OPTS)

run-xrt: prepare-data $(PROJECT)
	LD_LIBRARY_PATH=$(XILINX_XRT)/lib:$(VORTEX_RT_PATH):$(LD_LIBRARY_PATH) VORTEX_DRIVER=xrt ./$(PROJECT) $(OPTS)

clean:
	rm -rf $(PROJECT) *.o *.log .depend
	rm -f images weights layers 2>/dev/null; true

ifneq ($(MAKECMDGOALS),clean)
    -include .depend
endif
