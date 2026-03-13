BENCH_ROOT := $(realpath ../..)
include $(BENCH_ROOT)/config.mk

PROJECT := cnn

SRCS := \
    main.cpp \
    cnn_pipeline.cpp \
    layers/conv_gpu.cpp \
    layers/pool_gpu.cpp \
    layers/relu_gpu.cpp \
    layers/fc_cpu.cpp

OPTS ?= -n64

include $(BENCH_ROOT)/common/pipeline.mk
