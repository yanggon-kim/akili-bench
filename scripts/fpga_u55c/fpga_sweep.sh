#!/bin/bash
# usage: fpga_sweep.sh <outdir> <small|full>
# Builds and runs akili benchmarks on the U55C (bitstream wgspdxa16w_fullk_lsu1_nobal).
# Env: WL workloads, ML modes (tcu tcu_sp tcu_dxa tcu_sp_dxa), XEX extra WMMA
# flags, WGD/WGS extra dense/sparse WGMMA flags, OPOVR run options, LB llama
# batch, SKIP_RT=1 reuse the built runtime, TMO per-run timeout (s).
# Waits until no other user holds the card and reprograms it only when free.
OUT=$(realpath -m $1); SIZE=$2; mkdir -p $OUT
V=${VORTEX_HOME:-$(realpath "$(dirname "$0")/../../..")}; B=$V/akili-bench/benchmarks
export XILINX_XRT=/opt/xilinx/xrt PATH=/opt/xilinx/xrt/bin:$PATH
BIN=$V/hw/syn/xilinx/xrt/wgspdxa16w_fullk_lsu1_nobal_xilinx_u55c_gen3x16_xdma_3_202210_1_hw/bin
UUID=$(xclbinutil --info -i $BIN/vortex_afu.xclbin | awk '/UUID \(xclbin\)/{print toupper($3); exit}')
BDF=0000:3d:00.1; ME=$(id -u)
HW="-DVX_CFG_NUM_THREADS=8 -DVX_CFG_NUM_WARPS=16 -DVX_CFG_ISSUE_WIDTH=4 -DVX_CFG_NUM_LSU_BLOCKS=1 -DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_EXT_DXA_ENABLE -DVX_CFG_TCU_WGMMA_ENABLE -DVX_CFG_TCU_SPARSE_ENABLE -DVX_CFG_TCU_FEDP2K -DVX_CFG_TCU_USE_DSP=1 -DVX_CFG_TCU_SPARSE_FULLK"
K="-DITYPE=fp16 -DOTYPE=fp32"; WG="-DWGMMA_SS -DWGMMA_NRC=32 -DDEEP_BUF -DB_KMAJOR"
WGD=${WGD:--DPIPE_N=2 -DKB=4}; WGS=${WGS:--DPIPE_N=2 -DKB=2}
M1B=${LLAMA_1B_DIR:-$HOME/models/llama-3.2-1b}
if [ "$SIZE" = full ]; then
declare -A OP=( [acccnn]="-c 128 -o 128 -h 64 -w 64 -s 3" [attn]="-n 512 -d 3072" [flash]="-n 512 -d 3072" [NeRF]="-r 512 -s 64"
  [llama2]="$M1B/llama3_1b.bin -z $M1B/llama3_1b_tokenizer.bin -n 16 -v 1" [llama2_prefill]="$M1B/llama3_1b.bin -L 128 -v 1" )
else
declare -A OP=( [acccnn]="-c 8 -o 32 -h 8 -w 8 -s 3" [attn]="-n 64 -d 64" [flash]="-n 64 -d 64" [NeRF]="-r 32 -s 8"
  [llama2]="$B/akili_llama2/data/stories15M.bin -z $B/akili_llama2/data/tokenizer.bin -n 4 -v 1" [llama2_prefill]="$B/akili_llama2/data/stories15M.bin -L 32 -v 1" )
fi
others() { ps -eo uid=,cmd= | awk -v me=$ME '$1!=me && /VORTEX_DRIVER=xrt|run-xrt|xrt-smi program|xbutil program/' ; }
wait_free() {  # card free of other users for 60 s straight
  local quiet=0; while [ $quiet -lt 60 ]; do if [ -n "$(others)" ]; then quiet=0; else quiet=$((quiet+10)); fi; sleep 10; done; }
loaded() { timeout 60 xrt-smi examine -d $BDF -r dynamic-regions 2>/dev/null | awk '/Xclbin UUID/{print toupper($3)}'; }
ensure_ours() {
  while true; do
    wait_free
    [ "$(loaded)" = "$UUID" ] && return 0
    echo "$(date +%H:%M) programming card"
    # XRT refuses (EBUSY) while another process holds the card, so this
    # cannot preempt a running job; on refusal, back off and re-wait.
    if ! timeout 300 xrt-smi program -d $BDF -u $BIN/vortex_afu.xclbin >$OUT/_program.log 2>&1; then
      echo "$(date +%H:%M) card busy, waiting"; sleep 30
    fi
    [ "$(loaded)" = "$UUID" ] && return 0
  done; }
echo "$(date +%H:%M) waiting for card"; wait_free
if [ -z "${SKIP_RT:-}" ]; then
rm -f $V/sw/runtime/libvortex.so $V/sw/runtime/libvortex-xrt.so
CONFIGS="$HW" make -C $V/sw/runtime/stub DESTDIR=$V/sw/runtime > $OUT/_rt_stub.log 2>&1 || { echo "stub build failed"; exit 1; }
CONFIGS="$HW" TARGET=hw make -C $V/sw/runtime/xrt DESTDIR=$V/sw/runtime > $OUT/_rt_xrt.log 2>&1 || { echo "xrt runtime build failed"; tail -3 $OUT/_rt_xrt.log; exit 1; }
echo "$(date +%H:%M) runtime built"
fi
for w in ${WL:-acccnn attn flash NeRF llama2_prefill llama2}; do
  for m in ${ML:-tcu tcu_sp tcu_dxa tcu_sp_dxa}; do
    d=akili_${w}_${m}; ex="$K ${XEX:-}"; case $m in tcu_dxa) ex="$K $WG $WGD";; tcu_sp_dxa) ex="$K $WG $WGS";; esac
    case $w in llama2*) ex="$ex -DLLAMA_BATCH=${LB:-8}";; esac
    P=$(sed -n "s/^PROJECT *:= *//p" $B/$d/Makefile)
    (cd $B/$d && make clean >/dev/null 2>&1; make CONFIGS="$HW $ex" $P kernel.vxbin > $OUT/$d.build.log 2>&1) || { echo "$d BUILD FAILED"; continue; }
    ensure_ours
    t0=$(date +%s)
    (cd $B/$d && timeout ${TMO:-3600} make CONFIGS="$HW $ex" FPGA_BIN_DIR=$BIN XRT_DEVICE_INDEX=0 TARGET=hw OPTS="${OPOVR:-${OP[$w]}}" run-xrt > $OUT/$d.log 2>&1); rc=$?
    post=$(loaded); [ "$post" != "$UUID" ] && note="CARD-CHANGED($post)" || note=""
    printf "%s %-34s rc=%-3s %4ss %s %s | %s\n" "$(date +%H:%M)" $d $rc $(( $(date +%s)-t0 )) "$note" "$(grep -aoiE 'PASSED!?|FAILED!?|[0-9]+ errors?|mismatch' $OUT/$d.log | tail -1)" "$(grep -aiE -m1 'error:|abort|segmentation|timeout|terminate' $OUT/$d.log | cut -c1-120)"
  done
done
echo FPGA_ALL_DONE
