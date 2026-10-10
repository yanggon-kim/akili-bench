#!/bin/bash
# Regenerates every row of RESULTS_fpga_u55c.md's results table.
# usage: reproduce.sh <outdir> [row...]   rows: acccnn attn llama nerf fa moe (default all)
set -u
cd "$(dirname "$0")"
OUT=$(realpath -m "${1:-out}"); shift || true
ROWS=${*:-acccnn attn llama nerf fa moe}
M1B=${LLAMA_1B_DIR:-$HOME/models/llama-3.2-1b}
WG="-DPIPE_N=2 -DKB=4"
sweep() { local dir=$1; shift; env "$@" SKIP_RT=${SKIP_RT:-} TMO=${TMO:-7200} ./fpga_sweep.sh "$OUT/$dir" full > "$OUT/$dir.out" 2>&1; }
mkdir -p "$OUT"
for row in $ROWS; do
  case $row in
    acccnn) sweep acccnn WL=acccnn WGD="$WG" WGS="$WG" ;;
    attn)   sweep attn WL=attn WGD="$WG" WGS="$WG" ;;
    llama)
      L="$M1B/llama3_1b.bin -z $M1B/llama3_1b_tokenizer.bin -n 8 -s 42 -v 1 -P 128"
      sweep llama WL=llama2_e2e LB=32 OPOVR="$L" WGD="$WG" WGS="$WG" ;;
    nerf)
      sweep nerf_wmma WL=NeRF ML="tcu tcu_sp" XEX="-DAKILI_NERF_TILEDB"
      sweep nerf_wgmma WL=NeRF ML="tcu_dxa tcu_sp_dxa" WGD="$WG" WGS="$WG" ;;
    fa)
      R="-DFA_SM_RELOAD=2 -DFA_SMV2"
      W="$WG -DFA_REGSM -DFA_SMV2 -DFA_BC=16 -DFA_OLMEM -DFA_XCHG_LMEM -DFA_XCHG_NOFENCE -DFA_SUB=2 -DFA_ONE_BAR"
      sweep fa_wmma WL=fa ML=tcu OPOVR="-n 4096" XEX="$R -DFA_BR=8 -DFA_BC=64"
      sweep fa_wmma_sp WL=fa ML=tcu_sp OPOVR="-n 4096" XEX="$R -DFA_BR=16 -DFA_BC=32"
      sweep fa_wgmma WL=fa ML="tcu_dxa tcu_sp_dxa" OPOVR="-n 4096" WGD="$W" WGS="$W" ;;
    moe)    sweep moe WL=moe OPOVR="-t 512 -d 2048 -f 2048 -e 8" WGD="$WG" WGS="$WG" ;;
    *) echo "unknown row: $row"; exit 1 ;;
  esac
done
grep -rahE "AKILI_PERF|^E2E_(PREFILL|DECODE)" "$OUT" --include='*.log' -H | sed "s|$OUT/||"
