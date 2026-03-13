# Vortex Benchmarks — Top-level Makefile
#
# Adding a new benchmark:
#   1. Create a directory under benchmarks/<name>/
#   2. Add a Makefile that includes common/common.mk or common/pipeline.mk
#   3. That's it — it will be auto-discovered by this Makefile
#
# Usage:
#   make                    — build all benchmarks
#   make flash              — build just flash attention
#   make run-flash          — build and run flash on simx
#   make run-llama2         — build and run llama2 on simx
#   make cnn_kernels/conv   — build a specific kernel microbenchmark
#   make clean              — clean all

# Auto-discover all benchmarks (any directory under benchmarks/ with a Makefile)
BENCHMARKS := $(sort $(patsubst benchmarks/%/Makefile,%, \
    $(shell find benchmarks -name Makefile -not -path benchmarks/Makefile)))

.PHONY: all clean list $(BENCHMARKS)

all: $(BENCHMARKS)

# Build a specific benchmark
$(BENCHMARKS):
	@echo "==== Building $@ ===="
	$(MAKE) -C benchmarks/$@

# Run a specific benchmark on simx
run-%:
	$(MAKE) -C benchmarks/$* run-simx

# Clean a specific benchmark
clean-%:
	$(MAKE) -C benchmarks/$* clean

# Clean all benchmarks
clean:
	@for b in $(BENCHMARKS); do \
		echo "==== Cleaning $$b ===="; \
		$(MAKE) -C benchmarks/$$b clean; \
	done

# List all discovered benchmarks
list:
	@echo "Available benchmarks:"
	@for b in $(BENCHMARKS); do echo "  $$b"; done
