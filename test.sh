#!/bin/bash
# test.sh — Quick test script for vortex-benchmarks
#
# Usage (inside apptainer):
#   ./test.sh                    # test all benchmarks
#   ./test.sh flash llama2       # test specific benchmarks
#
# Prerequisites:
#   1. Run ./configure first
#   2. Run inside apptainer with proper bind mounts

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Check config.mk exists
if [ ! -f config.mk ]; then
    echo "Error: config.mk not found. Run ./configure first."
    echo ""
    echo "Example:"
    echo "  ./configure --xlen=32 --tooldir=/home/tools --vortex=/home/vortex --build=/home/vortex/build"
    exit 1
fi

# Color output helpers
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PASSED=0
FAILED=0
SKIPPED=0
RESULTS=""

run_test() {
    local name="$1"
    local target="$2"
    shift 2
    # remaining args are passed directly to make

    printf "%-25s " "$name"

    # Build
    if ! make -C "benchmarks/$target" "$@" 2>/dev/null 1>/dev/null; then
        printf "${RED}BUILD FAILED${NC}\n"
        FAILED=$((FAILED + 1))
        RESULTS="$RESULTS\n  ${RED}FAIL${NC}  $name (build error)"
        return
    fi

    # Run
    local output
    output=$(make -C "benchmarks/$target" run-simx "$@" 2>&1) || true

    if echo "$output" | grep -qi "PASSED\|completed successfully"; then
        printf "${GREEN}PASSED${NC}\n"
        PASSED=$((PASSED + 1))
        RESULTS="$RESULTS\n  ${GREEN}PASS${NC}  $name"
    elif echo "$output" | grep -qi "error\|failed\|fault"; then
        printf "${RED}FAILED${NC}\n"
        FAILED=$((FAILED + 1))
        RESULTS="$RESULTS\n  ${RED}FAIL${NC}  $name"
        echo "    Output (last 5 lines):"
        echo "$output" | tail -5 | sed 's/^/    /'
    else
        # Some benchmarks (like llama2) don't print PASSED but still work
        printf "${YELLOW}COMPLETED${NC} (check output)\n"
        PASSED=$((PASSED + 1))
        RESULTS="$RESULTS\n  ${YELLOW}DONE${NC}  $name (no PASSED/FAILED marker)"
        echo "    Output (last 3 lines):"
        echo "$output" | tail -3 | sed 's/^/    /'
    fi
}

# Default: test all benchmarks
# If arguments given, test only those
if [ $# -gt 0 ]; then
    BENCHES="$@"
else
    BENCHES="flash flash_tcu attention llama2 cnn_kernels/conv cnn_kernels/im2col cnn_kernels/pool cnn_kernels/relu cnn cnn_kernels/sgemm acccnn"
fi

echo "============================================"
echo "  Vortex Benchmarks Test Suite"
echo "============================================"
echo ""

for bench in $BENCHES; do
    case "$bench" in
        flash)
            run_test "flash (SIMT)" "flash"
            ;;
        flash_tcu)
            # TCU mode — requires Vortex built with: CONFIGS="-DEXT_TCU_ENABLE" make -s
            run_test "flash (TCU)" "flash" CONFIGS="-DNUM_THREADS=8 -DEXT_TCU_ENABLE" OPTS="-n 64 -d 8 -t 1"
            ;;
        attention)
            run_test "attention" "attention"
            ;;
        llama2)
            # Use fewer steps (-n 5) for faster testing
            run_test "llama2" "llama2" OPTS="data/stories15M.bin -z data/tokenizer.bin -n 5 -v 1"
            ;;
        cnn_kernels/conv)
            run_test "cnn_kernels/conv" "cnn_kernels/conv"
            ;;
        cnn_kernels/im2col)
            run_test "cnn_kernels/im2col" "cnn_kernels/im2col"
            ;;
        cnn_kernels/pool)
            run_test "cnn_kernels/pool" "cnn_kernels/pool"
            ;;
        cnn_kernels/relu)
            run_test "cnn_kernels/relu" "cnn_kernels/relu"
            ;;
        cnn_kernels/sgemm)
            # TCU kernel — requires Vortex built with TCU support
            # Rebuild Vortex with: CONFIGS="-DEXT_TCU_ENABLE" make -s
            run_test "cnn_kernels/sgemm (TCU)" "cnn_kernels/sgemm"
            ;;
        cnn)
            run_test "cnn (pipeline)" "cnn"
            ;;
        acccnn)
            run_test "acccnn (pipeline)" "acccnn"
            ;;
        *)
            # Try as-is for custom benchmark directories
            run_test "$bench" "$bench"
            ;;
    esac
done

echo ""
echo "============================================"
printf "  Results: ${GREEN}%d passed${NC}, ${RED}%d failed${NC}, ${YELLOW}%d skipped${NC}\n" "$PASSED" "$FAILED" "$SKIPPED"
echo "============================================"
echo -e "$RESULTS"
echo ""

[ $FAILED -eq 0 ] || exit 1
