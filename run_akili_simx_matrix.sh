#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
DEFAULT_BENCH_DIR="$SCRIPT_DIR"
DEFAULT_VORTEX_DIR="${HOME}/vortex"
DEFAULT_TOOLDIR="${HOME}/tools"
DEFAULT_PROFILING_CLASS=0
DEFAULT_L2_CACHE_SIZE=4194304
DEFAULT_L2_NUM_WAYS=8
DEFAULT_PLATFORM_MEMORY_BANKS=2
DEFAULT_ISSUE_WIDTH=4
DEFAULT_NUM_LSU_BLOCKS=2
DEFAULT_NUM_WARPS=16
DEFAULT_DCACHE_SIZE=65536
DEFAULT_DCACHE_NUM_WAYS=8
DEFAULT_L2_ENABLED=1
DEFAULT_BENCH_BUILD_LOCK="/tmp/akili_bench_build_${USER}.lock"
DEFAULT_TIMEOUT_SECS=1800
DEFAULT_PROFILE="sanity"
DEFAULT_THREADS_CSV="4,8,16,32"
DEFAULT_LLAMA_TOKENS=5
DEFAULT_LLAMA_TEMPERATURE=0
DEFAULT_RESULTS_ROOT="${HOME}/results"

timestamp_tag() {
  date '+%Y%m%d_%H%M%S'
}

timestamp_human() {
  date '+%Y-%m-%d %H:%M:%S'
}

log() {
  printf '[%s] %s\n' "$(timestamp_human)" "$*"
}

die() {
  log "ERROR: $*"
  exit 1
}

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Run all top-level akili_* benchmarks in simx across a configurable thread sweep.

Defaults:
  - benchmarks: every top-level benchmarks/akili_* directory
  - threads:    ${DEFAULT_THREADS_CSV}
  - profile:    ${DEFAULT_PROFILE}
  - llama tokens (sanity profile): ${DEFAULT_LLAMA_TOKENS}

Options:
  --threads LIST          Comma-separated thread counts (default: ${DEFAULT_THREADS_CSV})
  --benchmarks LIST       Comma-separated benchmark names to run
  --profile NAME          sanity or makefile (default: ${DEFAULT_PROFILE})
  --tokens N              Llama token count for the sanity profile (default: ${DEFAULT_LLAMA_TOKENS})
  --temperature N         Llama temperature for the sanity profile (default: ${DEFAULT_LLAMA_TEMPERATURE})
  --attn-opts STR         Override raw args for akili_attn*
  --flash-opts STR        Override raw args for akili_flash*
  --cnn-opts STR          Override raw args for akili_cnn / akili_acccnn_tcu*
  --nerf-opts STR         Override raw args for akili_NeRF*
  --llama-opts STR        Override raw args for akili_llama2*
  --benchmark-opts SPEC   Per-benchmark override, benchmark=args (repeatable)
  --output-dir PATH       Output directory
  --timeout-secs N        Per-benchmark timeout in seconds (default: ${DEFAULT_TIMEOUT_SECS})
  --jobs N                Parallel build jobs for make
  --cores N               NUM_CORES for Vortex builds (default: 4)
  --profiling-class N     VORTEX_PROFILING level (default: ${DEFAULT_PROFILING_CLASS})
  --issue-width N         ISSUE_WIDTH for Vortex builds (default: ${DEFAULT_ISSUE_WIDTH})
  --lsu-blocks N          NUM_LSU_BLOCKS for Vortex builds (default: ${DEFAULT_NUM_LSU_BLOCKS})
  --warps N               NUM_WARPS for Vortex builds (default: ${DEFAULT_NUM_WARPS})
  --disable-l2            Disable L2
  --l2-size BYTES         L2 size for Vortex builds (default: ${DEFAULT_L2_CACHE_SIZE})
  --l2-ways N             L2 associativity (default: ${DEFAULT_L2_NUM_WAYS})
  --mem-banks N           PLATFORM_MEMORY_NUM_BANKS (default: ${DEFAULT_PLATFORM_MEMORY_BANKS})
  --dcache-size BYTES     DCACHE_SIZE (default: ${DEFAULT_DCACHE_SIZE})
  --dcache-ways N         DCACHE_NUM_WAYS (default: ${DEFAULT_DCACHE_NUM_WAYS})
  --vortex-dir PATH       Vortex repository path (default: ~/vortex)
  --bench-dir PATH        akili-bench repository path (default: script dir)
  --tooldir PATH          Toolchain directory (default: ~/tools)
  --keep-going            Continue after failures instead of exiting nonzero
  --list                  Print discovered akili_* benchmarks and exit
  --dry-run               Print the selected matrix and exit
  -h, --help              Show this help

Examples:
  $(basename "$0")
  $(basename "$0") --threads 8 --benchmarks akili_attn,akili_flash_tcu,akili_llama2
  $(basename "$0") --threads 4,8 --cnn-opts "-c 32 -o 32 -h 32 -w 32 -s 3"
  $(basename "$0") --threads 8 --nerf-opts "-r 32 -s 8"
  $(basename "$0") --benchmark-opts akili_flash_tcu="-n 64 -d 512"
EOF
}

run_with_bench_build_lock() {
  local lock_path="$1"
  shift

  (
    flock 9
    "$@"
  ) 9>"$lock_path"
}

default_jobs() {
  local jobs
  jobs=$(getconf _NPROCESSORS_ONLN 2>/dev/null || true)
  if [[ -z "$jobs" || ! "$jobs" =~ ^[0-9]+$ || "$jobs" -lt 1 ]]; then
    jobs=8
  fi
  printf '%s\n' "$jobs"
}

require_file() {
  local path="$1"
  [[ -f "$path" ]] || die "Missing required file: $path"
}

require_command() {
  local cmd="$1"
  command -v "$cmd" >/dev/null 2>&1 || die "Missing required command: $cmd"
}

parse_csv_list() {
  local csv="$1"
  local -n out_ref="$2"
  local item

  IFS=',' read -r -a out_ref <<< "$csv"
  if [[ "${#out_ref[@]}" -eq 0 ]]; then
    die "Expected a non-empty comma-separated list"
  fi

  for item in "${!out_ref[@]}"; do
    out_ref[$item]=$(printf '%s' "${out_ref[$item]}" | xargs)
    [[ -n "${out_ref[$item]}" ]] || die "Invalid empty item in comma-separated list: $csv"
  done
}

parse_threads_csv() {
  local csv="$1"
  local -n out_ref="$2"
  local -a parsed=()
  local item

  parse_csv_list "$csv" parsed
  out_ref=("${parsed[@]}")
  for item in "${out_ref[@]}"; do
    [[ "$item" =~ ^[0-9]+$ ]] || die "Invalid thread count: $item"
    (( item > 0 )) || die "Thread count must be positive: $item"
  done
}

csv_escape() {
  local value="${1-}"
  value=${value//$'\r'/ }
  value=${value//$'\n'/ }
  value=${value//\"/\"\"}
  printf '"%s"' "$value"
}

append_csv_row() {
  local csv_file="$1"
  shift
  local first=1
  local value

  for value in "$@"; do
    if (( first )); then
      first=0
    else
      printf ',' >> "$csv_file"
    fi
    csv_escape "$value" >> "$csv_file"
  done
  printf '\n' >> "$csv_file"
}

extract_tok_s() {
  local log_file="$1"
  awk '
    match($0, /achieved tok\/s:[[:space:]]*([0-9.+-][0-9.eE+-]*)/, m) { value = m[1] }
    END { if (value != "") print value }
  ' "$log_file"
}

extract_perf_triplet() {
  local log_file="$1"
  awk '
    match($0, /PERF: instrs=([0-9]+), cycles=([0-9]+), IPC=([0-9.]+)/, m) {
      instrs = m[1]
      cycles = m[2]
      ipc = m[3]
    }
    END {
      if (instrs != "") {
        printf "%s\t%s\t%s\n", instrs, cycles, ipc
      }
    }
  ' "$log_file"
}

extract_run_time_sec() {
  local log_file="$1"
  awk '
    match($0, /run_time_sec=([0-9]+(\.[0-9]+)?)/, m) { value = m[1] }
    END { if (value != "") print value }
  ' "$log_file"
}

extract_generated_text() {
  local log_file="$1"
  awk '
    {
      lines[NR] = $0
      if (index($0, "achieved tok/s:")) {
        target = NR
      }
      if (index($0, "PERF: instrs=")) {
        perf_target = NR
      }
    }
    END {
      if (target) {
        line = lines[target]
        sub(/achieved tok\/s:.*/, "", line)
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", line)
        if (length(line) > 0) {
          print line
          exit
        }
        stop = target - 1
      } else if (perf_target) {
        stop = perf_target - 1
      } else {
        exit
      }

      for (i = stop; i >= 1; --i) {
        cand = lines[i]
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", cand)
        if (cand == "") {
          continue
        }
        if (cand ~ /^(PERF:|\[[0-9]{4}-[0-9]{2}-[0-9]{2}|make: |Running:|CONFIGS=)/) {
          continue
        }
        if (cand ~ /^(reusing bank|freeing bank|allocating bank|deallocating bank)/) {
          continue
        }
        print cand
        exit
      }
    }
  ' "$log_file"
}

extract_total_kcyc() {
  local log_file="$1"
  awk '
    match($0, /KCYC\[[^]]+\]:[[:space:]]*([0-9]+)/, m) { sum += m[1]; found = 1 }
    END { if (found) print sum }
  ' "$log_file"
}

extract_makefile_opts() {
  local makefile="$1"
  awk '
    /^[[:space:]]*OPTS[[:space:]]*\?=/ {
      sub(/^[[:space:]]*OPTS[[:space:]]*\?=[[:space:]]*/, "", $0)
      print
      exit
    }
  ' "$makefile"
}

extract_makefile_project() {
  local makefile="$1"
  awk '
    /^[[:space:]]*PROJECT[[:space:]]*:?=/ {
      sub(/^[[:space:]]*PROJECT[[:space:]]*:?=[[:space:]]*/, "", $0)
      print
      exit
    }
  ' "$makefile"
}

split_shell_words() {
  local raw="$1"
  local -n out_ref="$2"

  out_ref=()
  if [[ -n "$raw" ]]; then
    # shellcheck disable=SC2206
    out_ref=($raw)
  fi
}

remove_trace_artifacts() {
  local target="$1"

  [[ -e "$target" ]] || return 0

  find "$target" \
    \( -type d -name trace \
    -o -type f -name 'trace' \
    -o -type f -name 'trace.*' \
    -o -type f -name '*.vcd' \
    -o -type f -name '*.saif' \
    -o -type f -name 'simx_trace*' \) \
    -exec rm -rf {} +
}

source_env() {
  export TOOLDIR="$TOOLDIR_PATH"
  export CCACHE_DISABLE=1
  unset CCACHE_TEMPDIR || true
  # shellcheck disable=SC1091
  source "$VORTEX_DIR/ci/toolchain_env.sh"
}

append_build_config() {
  local current="$1"
  local option="$2"

  if [[ -n "$current" ]]; then
    printf '%s %s\n' "$current" "$option"
  else
    printf '%s\n' "$option"
  fi
}

ensure_build_tree() {
  local build_dir="$1"
  local build_configs="$2"
  local build_log="$3"

  mkdir -p "$build_dir"
  : > "$build_log"

  if [[ ! -f "$build_dir/config.mk" ]]; then
    log "Configuring Vortex build tree: $build_dir"
    (
      cd "$build_dir"
      "$VORTEX_DIR/configure" --xlen=32 --tooldir="$TOOLDIR_PATH"
    ) >> "$build_log" 2>&1
  fi

  log "Building SIMX runtime for $(basename "$build_dir")"
  make -C "$build_dir/hw" config CONFIGS="$build_configs" >> "$build_log" 2>&1
  make -C "$build_dir/kernel" -j"$JOBS" >> "$build_log" 2>&1
  make -C "$build_dir/runtime/stub" -j"$JOBS" >> "$build_log" 2>&1
  make -C "$build_dir/runtime/simx" CONFIGS="$build_configs" -j"$JOBS" >> "$build_log" 2>&1
}

extract_build_num_threads() {
  local build_dir="$1"
  local stamp="$build_dir/runtime/simx_rt_config.stamp"

  [[ -f "$stamp" ]] || return 1
  grep -o -- '-DNUM_THREADS=[0-9]*' "$stamp" | head -1 | cut -d= -f2
}

discover_benchmarks() {
  local benchmark

  for benchmark in "${SUPPORTED_BENCHMARKS[@]}"; do
    if [[ -d "$BENCH_DIR/benchmarks/$benchmark" ]]; then
      printf '%s\n' "$benchmark"
    fi
  done
}

benchmark_family() {
  case "$1" in
    akili_attn|akili_attn_tcu|akili_attn_tcu_sp|akili_attn_tcu_dxa|akili_attn_tcu_sp_dxa) printf 'attn\n' ;;
    akili_flash|akili_flash_tcu|akili_flash_tcu_sp|akili_flash_tcu_dxa|akili_flash_tcu_sp_dxa) printf 'flash\n' ;;
    akili_cnn|akili_acccnn_tcu|akili_acccnn_tcu_sp|akili_acccnn_tcu_dxa|akili_acccnn_tcu_sp_dxa) printf 'cnn\n' ;;
    akili_NeRF|akili_NeRF_tcu|akili_NeRF_tcu_sp|akili_NeRF_tcu_dxa|akili_NeRF_tcu_sp_dxa) printf 'nerf\n' ;;
    akili_llama2|akili_llama2_tcu|akili_llama2_tcu_sp|akili_llama2_tcu_dxa|akili_llama2_tcu_sp_dxa) printf 'llama\n' ;;
    akili_llama2_prefill|akili_llama2_prefill_tcu|akili_llama2_prefill_tcu_sp|akili_llama2_prefill_tcu_dxa|akili_llama2_prefill_tcu_sp_dxa) printf 'llama_prefill\n' ;;
    *) return 1 ;;
  esac
}

benchmark_mode() {
  case "$1" in
    *_tcu_sp_dxa) printf 'tcu_sparse_dxa\n' ;;
    *_tcu_dxa) printf 'tcu_dxa\n' ;;
    *_tcu_sp) printf 'tcu_sparse\n' ;;
    *_tcu) printf 'tcu\n' ;;
    *) printf 'simt\n' ;;
  esac
}

prepare_bench_workspace() {
  local workspace_dir="$1"
  shift
  local benchmark
  local have_llama_data=0

  rm -rf "$workspace_dir"
  mkdir -p "$workspace_dir/benchmarks"

  cp -a "$BENCH_DIR/config.mk" "$workspace_dir/"
  cp -a "$BENCH_DIR/common" "$workspace_dir/"

  for benchmark in "$@"; do
    cp -aL "$BENCH_DIR/benchmarks/$benchmark" "$workspace_dir/benchmarks/"
    if [[ "$benchmark" == akili_llama2* ]]; then
      have_llama_data=1
    fi
  done

  if (( have_llama_data )) && [[ ! -d "$workspace_dir/benchmarks/akili_llama2" ]]; then
    mkdir -p "$workspace_dir/benchmarks/akili_llama2"
    cp -a "$BENCH_DIR/benchmarks/akili_llama2/data" "$workspace_dir/benchmarks/akili_llama2/"
  fi

  remove_trace_artifacts "$workspace_dir"
}

resolve_default_opts() {
  local benchmark="$1"
  local workspace_dir="$2"
  local family raw base_llama_opts

  family=$(benchmark_family "$benchmark")

  if [[ -n "${BENCHMARK_OPTS[$benchmark]:-}" ]]; then
    printf '%s\n' "${BENCHMARK_OPTS[$benchmark]}"
    return
  fi

  case "$family" in
    attn)
      if [[ -n "$ATTN_OPTS_OVERRIDE" ]]; then
        printf '%s\n' "$ATTN_OPTS_OVERRIDE"
        return
      fi
      ;;
    flash)
      if [[ -n "$FLASH_OPTS_OVERRIDE" ]]; then
        printf '%s\n' "$FLASH_OPTS_OVERRIDE"
        return
      fi
      ;;
    cnn)
      if [[ -n "$CNN_OPTS_OVERRIDE" ]]; then
        printf '%s\n' "$CNN_OPTS_OVERRIDE"
        return
      fi
      ;;
    nerf)
      if [[ -n "$NERF_OPTS_OVERRIDE" ]]; then
        printf '%s\n' "$NERF_OPTS_OVERRIDE"
        return
      fi
      ;;
    llama|llama_prefill)
      case "$benchmark" in
        akili_llama2|akili_llama2_prefill)
          base_llama_opts="data/stories15M.bin -z data/tokenizer.bin"
          ;;
        akili_llama2_tcu|akili_llama2_tcu_sp|akili_llama2_tcu_dxa|akili_llama2_tcu_sp_dxa| \
        akili_llama2_prefill_tcu|akili_llama2_prefill_tcu_sp| \
        akili_llama2_prefill_tcu_dxa|akili_llama2_prefill_tcu_sp_dxa)
          base_llama_opts="../akili_llama2/data/stories15M.bin -z ../akili_llama2/data/tokenizer.bin"
          ;;
        *)
          die "No llama base args for benchmark: $benchmark"
          ;;
      esac

      if [[ -n "$LLAMA_OPTS_OVERRIDE" ]]; then
        if [[ "$LLAMA_OPTS_OVERRIDE" == -* ]]; then
          printf '%s %s\n' "$base_llama_opts" "$LLAMA_OPTS_OVERRIDE"
        else
          printf '%s\n' "$LLAMA_OPTS_OVERRIDE"
        fi
        return
      fi
      ;;
  esac

  if [[ "$PROFILE" == "makefile" ]]; then
    raw=$(extract_makefile_opts "$workspace_dir/benchmarks/$benchmark/Makefile")
    printf '%s\n' "$raw"
    return
  fi

  case "$benchmark" in
    akili_attn|akili_attn_tcu|akili_attn_tcu_sp|akili_attn_tcu_dxa|akili_attn_tcu_sp_dxa)
      printf '%s\n' '-n 16 -d 16'
      ;;
    akili_flash|akili_flash_tcu|akili_flash_tcu_sp|akili_flash_tcu_dxa|akili_flash_tcu_sp_dxa)
      printf '%s\n' '-n 16 -d 16'
      ;;
    akili_cnn|akili_acccnn_tcu|akili_acccnn_tcu_sp|akili_acccnn_tcu_dxa|akili_acccnn_tcu_sp_dxa)
      printf '%s\n' '-c 1 -o 8 -h 28 -w 28 -s 3'
      ;;
    akili_NeRF|akili_NeRF_tcu|akili_NeRF_tcu_sp|akili_NeRF_tcu_dxa|akili_NeRF_tcu_sp_dxa)
      printf '%s\n' '-r 32 -s 8'
      ;;
    akili_llama2|akili_llama2_prefill)
      printf '%s\n' "$base_llama_opts -n $TOKENS -t $TEMPERATURE -v 1"
      ;;
    akili_llama2_tcu|akili_llama2_tcu_sp|akili_llama2_tcu_dxa|akili_llama2_tcu_sp_dxa| \
    akili_llama2_prefill_tcu|akili_llama2_prefill_tcu_sp| \
    akili_llama2_prefill_tcu_dxa|akili_llama2_prefill_tcu_sp_dxa)
      printf '%s\n' "$base_llama_opts -n $TOKENS -t $TEMPERATURE -v 1"
      ;;
    *)
      die "No default opts for benchmark: $benchmark"
      ;;
  esac
}

run_single_benchmark() {
  local benchmark="$1"
  local workspace_dir="$2"
  local build_dir="$3"
  local nt="$4"
  local log_file="$5"

  local bench_path effective_opts run_time_sec start_time end_time
  local project_name
  local -a build_cmd run_cmd parsed_opts

  bench_path="$workspace_dir/benchmarks/$benchmark"
  project_name="$(extract_makefile_project "$bench_path/Makefile")"
  if [[ -z "$project_name" ]]; then
    project_name="$benchmark"
  fi
  effective_opts=$(resolve_default_opts "$benchmark" "$workspace_dir")
  split_shell_words "$effective_opts" parsed_opts

  build_cmd=(
    make -C "$bench_path" -B
    DEBUG=
    TRACE=
    VCD_OUTPUT=
    SAIF_OUTPUT=
    XLEN=32
    TOOLDIR="$TOOLDIR_PATH"
    VORTEX_HOME="$VORTEX_DIR"
    VORTEX_BUILD="$build_dir"
    NUM_THREADS="$nt"
    all
  )

  if [[ "$(benchmark_mode "$benchmark")" != "simt" ]]; then
    build_cmd+=(NUM_TCU_LANES="$nt")
  fi

  run_cmd=(
    env
    DEBUG=
    TRACE=
    VORTEX_TRACE=
    SIMX_BACKTRACE=
    VCD_OUTPUT=
    SAIF_OUTPUT=
    "LD_LIBRARY_PATH=$build_dir/runtime${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    "VORTEX_DRIVER=simx"
    "VORTEX_PROFILING=$PROFILING_CLASS"
    "./$project_name"
    "${parsed_opts[@]}"
  )

  {
    printf '[%s] benchmark=%s family=%s mode=%s threads=%s profile=%s\n' \
      "$(timestamp_human)" "$benchmark" "$(benchmark_family "$benchmark")" "$(benchmark_mode "$benchmark")" "$nt" "$PROFILE"
    printf '[%s] effective_opts=%s\n' "$(timestamp_human)" "$effective_opts"
    printf '[%s] build_cmd:' "$(timestamp_human)"
    printf ' %q' "${build_cmd[@]}"
    printf '\n'
    run_with_bench_build_lock "$BENCH_BUILD_LOCK" "${build_cmd[@]}"
    printf '[%s] run_cmd:' "$(timestamp_human)"
    printf ' %q' "${run_cmd[@]}"
    printf '\n'

    start_time="$(date +%s.%N)"
    (
      cd "$bench_path"
      timeout --signal=TERM --kill-after=30s "${TIMEOUT_SECS}s" "${run_cmd[@]}"
    )
    rc=$?
    end_time="$(date +%s.%N)"
    run_time_sec="$(awk -v start="$start_time" -v end="$end_time" 'BEGIN { printf "%.3f", end - start }')"
    printf '[%s] benchmark=%s run_time_sec=%s\n' "$(timestamp_human)" "$benchmark" "$run_time_sec"

    if [[ $rc -eq 124 ]]; then
      printf '[%s] benchmark=%s timed out after %ss\n' "$(timestamp_human)" "$benchmark" "$TIMEOUT_SECS"
    fi
    exit "$rc"
  } >> "$log_file" 2>&1
}

collect_result_fields() {
  local benchmark="$1"
  local workspace_dir="$2"
  local log_file="$3"
  local status="$4"
  local -n result_ref="$5"
  local effective_opts total_kcyc tok_s perf_triplet instrs cycles ipc run_time_sec generated_text speedup

  effective_opts=$(resolve_default_opts "$benchmark" "$workspace_dir")
  total_kcyc="$(extract_total_kcyc "$log_file" || true)"
  tok_s="$(extract_tok_s "$log_file" || true)"
  perf_triplet="$(extract_perf_triplet "$log_file" || true)"
  run_time_sec="$(extract_run_time_sec "$log_file" || true)"
  generated_text="$(extract_generated_text "$log_file" || true)"

  instrs=""
  cycles=""
  ipc=""
  if [[ -n "$perf_triplet" ]]; then
    IFS=$'\t' read -r instrs cycles ipc <<< "$perf_triplet"
  fi

  speedup=""
  if [[ "$status" == "passed" && -n "${BASELINE_LLAMA_CYCLES:-}" && "$benchmark" == akili_llama2* && -n "$cycles" ]]; then
    speedup="$(awk -v baseline="$BASELINE_LLAMA_CYCLES" -v current="$cycles" '
      BEGIN {
        if (baseline + 0 > 0 && current + 0 > 0) {
          printf "%.6f\n", baseline / current
        }
      }'
    )"
  fi

  result_ref=(
    "$effective_opts"
    "$tok_s"
    "$total_kcyc"
    "$instrs"
    "$cycles"
    "$ipc"
    "$run_time_sec"
    "$generated_text"
    "$speedup"
  )
}

validate_selected_benchmarks() {
  local benchmark
  local -A available_map=()

  for benchmark in "${DISCOVERED_BENCHMARKS[@]}"; do
    available_map["$benchmark"]=1
  done

  for benchmark in "${SELECTED_BENCHMARKS[@]}"; do
    [[ -n "${available_map[$benchmark]:-}" ]] || die "Unknown benchmark: $benchmark"
    benchmark_family "$benchmark" >/dev/null || die "Unsupported benchmark family mapping: $benchmark"
  done
}

run_nt_suite() {
  local nt="$1"
  local part_csv="$2"
  local summary_file="$3"
  local case_name nt_log_dir build_dir build_log build_configs workspace_dir dcache_banks
  local total_runs=0
  local passed_runs=0
  local failed_runs=0
  local benchmark log_file pid status wait_rc
  local baseline_cycles=""
  local -A pid_to_benchmark=()
  local -A bench_status=()
  local -A result_effective_opts=()
  local -A result_tok_s=()
  local -A result_total_kcyc=()
  local -A result_instrs=()
  local -A result_cycles=()
  local -A result_ipc=()
  local -A result_run_time=()
  local -A result_text=()
  local -A result_speedup=()

  case_name="${CORES}c_nt${nt}_akili"
  nt_log_dir="$OUTPUT_DIR/logs/${CORES}c_nt${nt}"
  build_dir="$OUTPUT_DIR/vortex_builds/$case_name"
  workspace_dir="$OUTPUT_DIR/workspaces/${CORES}c_nt${nt}"
  build_log="$nt_log_dir/build.log"
  dcache_banks="$nt"
  if (( dcache_banks > 16 )); then
    dcache_banks=16
  fi

  build_configs="-DNUM_CORES=$CORES -DNUM_THREADS=$nt -DEXT_TCU_ENABLE -DTCU_SPARSE_ENABLE -DEXT_DXA_ENABLE -DISSUE_WIDTH=$ISSUE_WIDTH -DNUM_LSU_BLOCKS=$NUM_LSU_BLOCKS -DNUM_WARPS=$NUM_WARPS -DPLATFORM_MEMORY_NUM_BANKS=$PLATFORM_MEMORY_BANKS -DDCACHE_SIZE=$DCACHE_SIZE -DDCACHE_NUM_WAYS=$DCACHE_NUM_WAYS -DDCACHE_NUM_BANKS=$dcache_banks -DDCACHE_WRITEBACK=0 -DL2_WRITEBACK=0"
  if (( L2_ENABLED )); then
    build_configs="$(append_build_config "$build_configs" "-DL2_ENABLE")"
    build_configs="$(append_build_config "$build_configs" "-DL2_CACHE_SIZE=$L2_CACHE_SIZE")"
    build_configs="$(append_build_config "$build_configs" "-DL2_NUM_WAYS=$L2_NUM_WAYS")"
  fi
  if (( PROFILING_CLASS > 0 )); then
    build_configs="$(append_build_config "$build_configs" "-DPERF_ENABLE")"
  fi

  mkdir -p "$nt_log_dir"
  : > "$part_csv"

  log "[nt=$nt] Preparing isolated benchmark workspace: $workspace_dir"
  prepare_bench_workspace "$workspace_dir" "${SELECTED_BENCHMARKS[@]}"

  if ! run_with_bench_build_lock "$BENCH_BUILD_LOCK" ensure_build_tree "$build_dir" "$build_configs" "$build_log"; then
    log "[nt=$nt] Vortex build failed"
    for benchmark in "${SELECTED_BENCHMARKS[@]}"; do
      append_csv_row \
        "$part_csv" \
        "$(timestamp_human)" \
        "$benchmark" \
        "$(benchmark_family "$benchmark")" \
        "$(benchmark_mode "$benchmark")" \
        "$CORES" \
        "$nt" \
        "$PROFILE" \
        "" \
        "build_failed" \
        "" \
        "" \
        "" \
        "" \
        "" \
        "" \
        "" \
        "" \
        "$nt_log_dir/${benchmark}.log"
    done
    printf 'TOTAL_RUNS=%s\nPASSED_RUNS=%s\nFAILED_RUNS=%s\n' \
      "${#SELECTED_BENCHMARKS[@]}" 0 "${#SELECTED_BENCHMARKS[@]}" > "$summary_file"
    return 0
  fi

  local built_nt=""
  built_nt="$(extract_build_num_threads "$build_dir" || true)"
  if [[ -z "$built_nt" || "$built_nt" != "$nt" ]]; then
    log "[nt=$nt] Build tree thread mismatch: expected $nt, found ${built_nt:-missing}"
    {
      printf '[%s] expected_num_threads=%s\n' "$(timestamp_human)" "$nt"
      printf '[%s] build_dir=%s\n' "$(timestamp_human)" "$build_dir"
      printf '[%s] simx_rt_config_num_threads=%s\n' "$(timestamp_human)" "${built_nt:-missing}"
      if [[ -f "$build_dir/runtime/simx_rt_config.stamp" ]]; then
        printf '[%s] simx_rt_config_stamp=%s\n' "$(timestamp_human)" "$build_dir/runtime/simx_rt_config.stamp"
      fi
    } >> "$build_log"
    for benchmark in "${SELECTED_BENCHMARKS[@]}"; do
      append_csv_row \
        "$part_csv" \
        "$(timestamp_human)" \
        "$benchmark" \
        "$(benchmark_family "$benchmark")" \
        "$(benchmark_mode "$benchmark")" \
        "$CORES" \
        "$nt" \
        "$PROFILE" \
        "" \
        "build_failed" \
        "" \
        "" \
        "" \
        "" \
        "" \
        "" \
        "" \
        "" \
        "$nt_log_dir/${benchmark}.log"
    done
    printf 'TOTAL_RUNS=%s\nPASSED_RUNS=%s\nFAILED_RUNS=%s\n' \
      "${#SELECTED_BENCHMARKS[@]}" 0 "${#SELECTED_BENCHMARKS[@]}" > "$summary_file"
    return 0
  fi

  log "[nt=$nt] Launching ${#SELECTED_BENCHMARKS[@]} benchmarks in parallel"
  for benchmark in "${SELECTED_BENCHMARKS[@]}"; do
    log_file="$nt_log_dir/${benchmark}.log"
    (
      run_single_benchmark "$benchmark" "$workspace_dir" "$build_dir" "$nt" "$log_file"
    ) &
    pid_to_benchmark[$!]="$benchmark"
  done

  for pid in "${!pid_to_benchmark[@]}"; do
    benchmark="${pid_to_benchmark[$pid]}"
    if wait "$pid"; then
      if [[ "$benchmark" == akili_llama2* ]]; then
        status="passed"
      elif grep -q "PASSED!" "$nt_log_dir/${benchmark}.log"; then
        status="passed"
      else
        status="failed"
      fi
    else
      wait_rc=$?
      if (( wait_rc == 124 )); then
        status="timeout"
      else
        status="failed"
      fi
    fi
    bench_status["$benchmark"]="$status"
    total_runs=$((total_runs + 1))
    if [[ "$status" == "passed" ]]; then
      passed_runs=$((passed_runs + 1))
    else
      failed_runs=$((failed_runs + 1))
    fi
  done

  if [[ -f "$nt_log_dir/akili_llama2.log" && "${bench_status[akili_llama2]:-}" == "passed" ]]; then
    baseline_cycles="$(extract_perf_triplet "$nt_log_dir/akili_llama2.log" | awk -F'\t' 'NF >= 2 { print $2 }' || true)"
  fi
  BASELINE_LLAMA_CYCLES="$baseline_cycles"

  for benchmark in "${SELECTED_BENCHMARKS[@]}"; do
    local -a parsed_fields=()
    collect_result_fields "$benchmark" "$workspace_dir" "$nt_log_dir/${benchmark}.log" "${bench_status[$benchmark]}" parsed_fields
    result_effective_opts["$benchmark"]="${parsed_fields[0]}"
    result_tok_s["$benchmark"]="${parsed_fields[1]}"
    result_total_kcyc["$benchmark"]="${parsed_fields[2]}"
    result_instrs["$benchmark"]="${parsed_fields[3]}"
    result_cycles["$benchmark"]="${parsed_fields[4]}"
    result_ipc["$benchmark"]="${parsed_fields[5]}"
    result_run_time["$benchmark"]="${parsed_fields[6]}"
    result_text["$benchmark"]="${parsed_fields[7]}"
    result_speedup["$benchmark"]="${parsed_fields[8]}"

    append_csv_row \
      "$part_csv" \
      "$(timestamp_human)" \
      "$benchmark" \
      "$(benchmark_family "$benchmark")" \
      "$(benchmark_mode "$benchmark")" \
      "$CORES" \
      "$nt" \
      "$PROFILE" \
      "${result_effective_opts[$benchmark]}" \
      "${bench_status[$benchmark]}" \
      "${result_tok_s[$benchmark]}" \
      "${result_total_kcyc[$benchmark]}" \
      "${result_instrs[$benchmark]}" \
      "${result_cycles[$benchmark]}" \
      "${result_ipc[$benchmark]}" \
      "${result_speedup[$benchmark]}" \
      "${result_run_time[$benchmark]}" \
      "${result_text[$benchmark]}" \
      "$nt_log_dir/${benchmark}.log"
  done

  printf 'TOTAL_RUNS=%s\nPASSED_RUNS=%s\nFAILED_RUNS=%s\n' \
    "$total_runs" "$passed_runs" "$failed_runs" > "$summary_file"

  remove_trace_artifacts "$build_dir"
  remove_trace_artifacts "$workspace_dir"

  if (( failed_runs > 0 )); then
    log "[nt=$nt] Completed with failures"
  else
    log "[nt=$nt] Completed successfully"
  fi
}

CORES=4
THREADS_CSV="$DEFAULT_THREADS_CSV"
TOKENS="$DEFAULT_LLAMA_TOKENS"
TEMPERATURE="$DEFAULT_LLAMA_TEMPERATURE"
PROFILE="$DEFAULT_PROFILE"
OUTPUT_DIR=""
JOBS="$(default_jobs)"
KEEP_GOING=0
PROFILING_CLASS="$DEFAULT_PROFILING_CLASS"
ISSUE_WIDTH="$DEFAULT_ISSUE_WIDTH"
NUM_LSU_BLOCKS="$DEFAULT_NUM_LSU_BLOCKS"
NUM_WARPS="$DEFAULT_NUM_WARPS"
L2_ENABLED="$DEFAULT_L2_ENABLED"
L2_CACHE_SIZE="$DEFAULT_L2_CACHE_SIZE"
L2_NUM_WAYS="$DEFAULT_L2_NUM_WAYS"
PLATFORM_MEMORY_BANKS="$DEFAULT_PLATFORM_MEMORY_BANKS"
DCACHE_SIZE="$DEFAULT_DCACHE_SIZE"
DCACHE_NUM_WAYS="$DEFAULT_DCACHE_NUM_WAYS"
BENCH_BUILD_LOCK="$DEFAULT_BENCH_BUILD_LOCK"
TIMEOUT_SECS="$DEFAULT_TIMEOUT_SECS"
VORTEX_DIR="$DEFAULT_VORTEX_DIR"
BENCH_DIR="$DEFAULT_BENCH_DIR"
TOOLDIR_PATH="$DEFAULT_TOOLDIR"
ATTN_OPTS_OVERRIDE=""
FLASH_OPTS_OVERRIDE=""
CNN_OPTS_OVERRIDE=""
NERF_OPTS_OVERRIDE=""
LLAMA_OPTS_OVERRIDE=""
LIST_ONLY=0
DRY_RUN=0
declare -a THREADS=()
declare -a DISCOVERED_BENCHMARKS=()
declare -a SELECTED_BENCHMARKS=()
declare -a SUPPORTED_BENCHMARKS=(
  akili_attn
  akili_attn_tcu
  akili_attn_tcu_sp
  akili_flash
  akili_flash_tcu
  akili_flash_tcu_sp
  akili_cnn
  akili_acccnn_tcu
  akili_acccnn_tcu_sp
  akili_NeRF
  akili_NeRF_tcu
  akili_NeRF_tcu_sp
  akili_llama2
  akili_llama2_tcu
  akili_llama2_tcu_sp
  akili_attn_tcu_dxa
  akili_attn_tcu_sp_dxa
  akili_flash_tcu_dxa
  akili_flash_tcu_sp_dxa
  akili_acccnn_tcu_dxa
  akili_acccnn_tcu_sp_dxa
  akili_NeRF_tcu_dxa
  akili_NeRF_tcu_sp_dxa
  akili_llama2_tcu_dxa
  akili_llama2_tcu_sp_dxa
  akili_llama2_prefill
  akili_llama2_prefill_tcu
  akili_llama2_prefill_tcu_sp
  akili_llama2_prefill_tcu_dxa
  akili_llama2_prefill_tcu_sp_dxa
)
declare -A BENCHMARK_OPTS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --threads)
      THREADS_CSV="$2"
      shift 2
      ;;
    --benchmarks)
      BENCHMARKS_CSV="$2"
      shift 2
      ;;
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --tokens)
      TOKENS="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --attn-opts)
      ATTN_OPTS_OVERRIDE="$2"
      shift 2
      ;;
    --flash-opts)
      FLASH_OPTS_OVERRIDE="$2"
      shift 2
      ;;
    --cnn-opts)
      CNN_OPTS_OVERRIDE="$2"
      shift 2
      ;;
    --nerf-opts)
      NERF_OPTS_OVERRIDE="$2"
      shift 2
      ;;
    --llama-opts)
      LLAMA_OPTS_OVERRIDE="$2"
      shift 2
      ;;
    --benchmark-opts)
      [[ "$2" == *=* ]] || die "--benchmark-opts expects benchmark=args"
      BENCHMARK_OPTS["${2%%=*}"]="${2#*=}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --timeout-secs)
      TIMEOUT_SECS="$2"
      shift 2
      ;;
    --jobs)
      JOBS="$2"
      shift 2
      ;;
    --cores)
      CORES="$2"
      shift 2
      ;;
    --profiling-class)
      PROFILING_CLASS="$2"
      shift 2
      ;;
    --issue-width)
      ISSUE_WIDTH="$2"
      shift 2
      ;;
    --lsu-blocks)
      NUM_LSU_BLOCKS="$2"
      shift 2
      ;;
    --warps)
      NUM_WARPS="$2"
      shift 2
      ;;
    --disable-l2)
      L2_ENABLED=0
      shift
      ;;
    --l2-size)
      L2_CACHE_SIZE="$2"
      shift 2
      ;;
    --l2-ways)
      L2_NUM_WAYS="$2"
      shift 2
      ;;
    --mem-banks)
      PLATFORM_MEMORY_BANKS="$2"
      shift 2
      ;;
    --dcache-size)
      DCACHE_SIZE="$2"
      shift 2
      ;;
    --dcache-ways)
      DCACHE_NUM_WAYS="$2"
      shift 2
      ;;
    --vortex-dir)
      VORTEX_DIR="$2"
      shift 2
      ;;
    --bench-dir)
      BENCH_DIR="$2"
      shift 2
      ;;
    --tooldir)
      TOOLDIR_PATH="$2"
      shift 2
      ;;
    --keep-going)
      KEEP_GOING=1
      shift
      ;;
    --list)
      LIST_ONLY=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      die "Unknown argument: $1"
      ;;
  esac
done

[[ "$CORES" =~ ^[0-9]+$ ]] || die "--cores must be a positive integer"
(( CORES > 0 )) || die "--cores must be positive"
[[ "$TOKENS" =~ ^[0-9]+$ ]] || die "--tokens must be a positive integer"
(( TOKENS > 0 )) || die "--tokens must be positive"
[[ "$TEMPERATURE" =~ ^-?[0-9]+$ ]] || die "--temperature must be an integer"
[[ "$JOBS" =~ ^[0-9]+$ ]] || die "--jobs must be a positive integer"
(( JOBS > 0 )) || die "--jobs must be positive"
[[ "$TIMEOUT_SECS" =~ ^[0-9]+$ ]] || die "--timeout-secs must be a positive integer"
(( TIMEOUT_SECS > 0 )) || die "--timeout-secs must be positive"
[[ "$PROFILING_CLASS" =~ ^[0-9]+$ ]] || die "--profiling-class must be a non-negative integer"
[[ "$ISSUE_WIDTH" =~ ^[0-9]+$ ]] || die "--issue-width must be a positive integer"
(( ISSUE_WIDTH > 0 )) || die "--issue-width must be positive"
[[ "$NUM_LSU_BLOCKS" =~ ^[0-9]+$ ]] || die "--lsu-blocks must be a positive integer"
(( NUM_LSU_BLOCKS > 0 )) || die "--lsu-blocks must be positive"
[[ "$NUM_WARPS" =~ ^[0-9]+$ ]] || die "--warps must be a positive integer"
(( NUM_WARPS > 0 )) || die "--warps must be positive"
[[ "$L2_CACHE_SIZE" =~ ^[0-9]+$ ]] || die "--l2-size must be a positive integer"
(( L2_CACHE_SIZE > 0 )) || die "--l2-size must be positive"
[[ "$L2_NUM_WAYS" =~ ^[0-9]+$ ]] || die "--l2-ways must be a positive integer"
(( L2_NUM_WAYS > 0 )) || die "--l2-ways must be positive"
[[ "$PLATFORM_MEMORY_BANKS" =~ ^[0-9]+$ ]] || die "--mem-banks must be a positive integer"
(( PLATFORM_MEMORY_BANKS > 0 )) || die "--mem-banks must be positive"
[[ "$DCACHE_SIZE" =~ ^[0-9]+$ ]] || die "--dcache-size must be a positive integer"
(( DCACHE_SIZE > 0 )) || die "--dcache-size must be positive"
[[ "$DCACHE_NUM_WAYS" =~ ^[0-9]+$ ]] || die "--dcache-ways must be a positive integer"
(( DCACHE_NUM_WAYS > 0 )) || die "--dcache-ways must be positive"
[[ "$PROFILE" == "sanity" || "$PROFILE" == "makefile" ]] || die "--profile must be sanity or makefile"

parse_threads_csv "$THREADS_CSV" THREADS

[[ -d "$VORTEX_DIR" ]] || die "Vortex directory does not exist: $VORTEX_DIR"
[[ -d "$BENCH_DIR" ]] || die "Bench directory does not exist: $BENCH_DIR"
require_file "$VORTEX_DIR/configure"
require_file "$VORTEX_DIR/ci/toolchain_env.sh"
require_file "$BENCH_DIR/config.mk"
require_file "$BENCH_DIR/common/common.mk"
require_command awk
require_command find
require_command flock
require_command make
require_command sort
require_command timeout

mapfile -t DISCOVERED_BENCHMARKS < <(discover_benchmarks)
(( ${#DISCOVERED_BENCHMARKS[@]} > 0 )) || die "No benchmarks matching benchmarks/akili_* were found"

if (( LIST_ONLY )); then
  printf '%s\n' "${DISCOVERED_BENCHMARKS[@]}"
  exit 0
fi

if [[ -n "${BENCHMARKS_CSV:-}" ]]; then
  parse_csv_list "$BENCHMARKS_CSV" SELECTED_BENCHMARKS
else
  SELECTED_BENCHMARKS=("${DISCOVERED_BENCHMARKS[@]}")
fi

validate_selected_benchmarks

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$DEFAULT_RESULTS_ROOT/akili_simx_$(timestamp_tag)"
fi

if (( DRY_RUN )); then
  printf 'output_dir=%s\n' "$OUTPUT_DIR"
  printf 'profile=%s\n' "$PROFILE"
  printf 'threads=%s\n' "$THREADS_CSV"
  printf 'benchmarks=%s\n' "${SELECTED_BENCHMARKS[*]}"
  for benchmark in "${SELECTED_BENCHMARKS[@]}"; do
    printf '%s\tfamily=%s\tmode=%s\targs=%s\n' \
      "$benchmark" "$(benchmark_family "$benchmark")" "$(benchmark_mode "$benchmark")" \
      "$(resolve_default_opts "$benchmark" "$BENCH_DIR")"
  done
  exit 0
fi

mkdir -p "$OUTPUT_DIR" "$OUTPUT_DIR/logs" "$OUTPUT_DIR/vortex_builds" "$OUTPUT_DIR/workspaces"
RESULTS_CSV="$OUTPUT_DIR/results.csv"
remove_trace_artifacts "$OUTPUT_DIR"

source_env

log "Output directory: $OUTPUT_DIR"
log "Profile: $PROFILE"
log "Threads: $THREADS_CSV"
log "Benchmarks: ${SELECTED_BENCHMARKS[*]}"

declare -A NT_TO_PID=()

for nt in "${THREADS[@]}"; do
  part_csv="$OUTPUT_DIR/results_nt${nt}.csv.part"
  summary_file="$OUTPUT_DIR/results_nt${nt}.summary"
  (
    run_nt_suite "$nt" "$part_csv" "$summary_file"
  ) &
  NT_TO_PID["$nt"]=$!
done

worker_error=0
for nt in "${THREADS[@]}"; do
  if ! wait "${NT_TO_PID[$nt]}"; then
    log "[nt=$nt] Worker exited unexpectedly"
    worker_error=1
  fi
done

cat > "$RESULTS_CSV" <<'EOF'
"timestamp","benchmark","family","mode","cores","threads","profile","effective_args","status","tok_s","total_kcyc","instrs","cycles","ipc","cycle_speedup_vs_llama2","run_time_sec","generated_text","log_file"
EOF

total_runs=0
passed_runs=0
failed_runs=0

for nt in "${THREADS[@]}"; do
  part_csv="$OUTPUT_DIR/results_nt${nt}.csv.part"
  summary_file="$OUTPUT_DIR/results_nt${nt}.summary"

  if [[ -f "$part_csv" ]]; then
    cat "$part_csv" >> "$RESULTS_CSV"
  fi
  if [[ -f "$summary_file" ]]; then
    nt_total="$(awk -F= '$1 == "TOTAL_RUNS" { print $2 }' "$summary_file")"
    nt_passed="$(awk -F= '$1 == "PASSED_RUNS" { print $2 }' "$summary_file")"
    nt_failed="$(awk -F= '$1 == "FAILED_RUNS" { print $2 }' "$summary_file")"
    total_runs=$((total_runs + ${nt_total:-0}))
    passed_runs=$((passed_runs + ${nt_passed:-0}))
    failed_runs=$((failed_runs + ${nt_failed:-0}))
  else
    log "[nt=$nt] Missing summary file: $summary_file"
    worker_error=1
  fi
done

remove_trace_artifacts "$OUTPUT_DIR"

log "Completed matrix run"
log "Results CSV: $RESULTS_CSV"
log "Passed: $passed_runs Failed: $failed_runs Total: $total_runs"

if (( worker_error != 0 || failed_runs != 0 )); then
  if (( KEEP_GOING )); then
    exit 0
  fi
  exit 1
fi
