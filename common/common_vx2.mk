# common/common_vx2.mk — Build rules for benchmarks written against the current
# Vortex tree (sw/ layout, -DVX_CFG_* configs projected from VX_config.toml).
#
# Reuses the configured Vortex build's own regression rules; only its root
# (normally derived from the test's location inside Vortex) is pinned to
# VORTEX_BUILD. A benchmark Makefile includes $(BENCH_ROOT)/config.mk and
# $(VORTEX_BUILD)/config.mk, defines PROJECT/SRCS/VX_SRCS, then this file.
# akili's config.mk points SW_COMMON_DIR at the old sim/common location.
override ROOT_DIR := $(VORTEX_BUILD)
override SW_COMMON_DIR := $(VORTEX_HOME)/sw/common
include $(VORTEX_BUILD)/tests/regression/common.mk
