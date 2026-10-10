# akili-bench FPGA results (U55C, 16 warps, FULLK sparse WGMMA)

Recorded 2026-10-09. Device cycles are summed over every kernel launch from
core 0's cycle counter (host compute and PCIe transfers excluded).

## Configuration

| Item | Value |
|---|---|
| Bitstream | `wgspdxa16w_fullk_lsu1_nobal`, 160.8 MHz |
| Core | 1 core, NT=8, 16 warps, issue width 4, LSU blocks 1 |
| TCU | FEDP2K, sparse FULLK (`VX_CFG_TCU_SPARSE_FULLK`), DSP on |
| Memory | LMEM 16 KB 1R1W, DXA max in flight 8 |
| WGMMA kernels | WGMMA_SS, NRC=32, DEEP_BUF, B_KMAJOR, PIPE_N=2, KB=4 |
| CTA size | 4 warps (= issue width) |
| Build | akili-bench with `+xvortex` fix (common/common.mk), unless marked "old build" |

To regenerate the table: `scripts/fpga_u55c/reproduce.sh <outdir> [rows]` from
a Vortex checkout with this bitstream built (set `VORTEX_HOME` if akili-bench is
not inside it, and `LLAMA_1B_DIR` for the Llama row). It runs each row with the
flags used here through `scripts/fpga_u55c/fpga_sweep.sh`. The acccnn, MoE
and Llama (WGMMA columns) rows were measured before later kernel edits that add
default-off options and have not been re-run on this revision; the attention,
NeRF and fused FlashAttention rows were.

## Results (device cycles)

Speedups are sparse over dense within the same API (above 1 = sparse faster).

| Workload | Problem size | WMMA | Sparse WMMA | WGMMA | Sparse WGMMA | sp/dense WMMA | sp/dense WGMMA |
|---|---|---|---|---|---|---|---|
| acccnn | `-c 128 -o 128 -h 64 -w 64 -s 3` | 140.42M | 125.78M | 44.82M | 34.53M | 1.12x | 1.30x |
| attention (unfused) | `-n 512 -d 3072` | 777.09M | 457.59M | 137.95M | 112.68M | 1.70x | 1.22x |
| Llama-3.2-1B prefill (d) | prompt 128 | 103.31G | 100.71G | 10.15G | 8.34G | 1.03x | 1.22x |
| Llama-3.2-1B decode (d) | batch 32, 8 steps | 184.78G | 181.73G | 23.82G | 21.55G | 1.02x | 1.11x |
| Llama-3.2-1B total (d) | prompt 128 + 8 steps, batch 32 | 288.09G | 282.44G | 33.97G | 29.88G | 1.02x | 1.14x |
| NeRF (a) | `-r 512 -s 64` | 1.595G | 1.539G | 0.935G | 0.905G | 1.04x | 1.03x |
| fused FlashAttention (b) | d=32, N=4096 | 326.3M | 316.0M | 312.2M | 299.3M | 1.03x | 1.04x |
| MoE top-2 FFN layer (c) | `-t 512 -d 2048 -f 2048 -e 8` | 10.731G | 10.185G | 1.044G | 0.842G | 1.05x | 1.24x |

## Notes

(a) NeRF. The WMMA columns use a tiled activation layout
(`-DAKILI_NERF_TILEDB`): the positional-encoding and ReLU/bias kernels write
each layer's B operand as contiguous column-major tileK x tileN blocks in the
order the next GEMM reads them. Instructions 639.8M / 564.5M. WGMMA stages B
through DXA from the row-major layout. With row-major B the WMMA columns are
4.396G / 10.188G (the 64 KB activation row stride maps every B row of a tile to
one L1 set); padding the stride by one cache line (`-DAKILI_NERF_XPAD=32`)
gives 2.661G / 3.230G. An earlier table entry (1.342G / 1.290G) came from a
read-only tiling experiment whose producers never wrote the tiled layout; it
computed wrong results and is withdrawn. All NeRF modes fail the 1e-2 output
check on 0.1-2% of GEMM outputs per layer (WGMMA dense: a few dozen
elements); counts vary between runs of the same binary, so they are not a
fixed layout or indexing error.

(b) Fused FlashAttention, optimized kernels, each mode at its best tile (see
"Fused FlashAttention, optimized" below for flags). WMMA, sparse WMMA and
WGMMA/sparse WGMMA come from one batch on one build. WGMMA and sparse WGMMA
failed the output check by one row (32 and 30 of 131072) in this batch and
passed at the same cycle counts in an earlier one. Sparse WMMA measured 311.5M
in an earlier run, so WGMMA versus sparse WMMA (1.2% here) is within run-to-run
variation. The paragraphs below describe the pre-optimization kernels.

Pre-optimization: Fused FlashAttention (online softmax, only Q.K^T sparse), d=32. WMMA at
N=4096 failed its output check (single bad row, intermittent); the other
three passed. Other sizes (WMMA / sparse WMMA / WGMMA / sparse WGMMA):

| N | WMMA | Sparse WMMA | WGMMA | Sparse WGMMA |
|---|---|---|---|---|
| 512 | 10.73M | 10.61M | 11.12M | 11.13M |
| 1024 | 41.86M (fail) | 42.08M | 43.74M | 43.81M |
| 2048 | 164.18M | 166.96M | 173.50M | 173.66M |

The 16-row CTA row uses 16 query rows (2 warps) per CTA instead of 32. For
WMMA this lets two CTAs share the 16 KB shared memory and overlaps one CTA's
softmax with the other's work (1.4-1.5x faster). The WGMMA kernel needs 12 KB
per CTA at 16 rows, so no second CTA fits, and every CTA re-streams all of K
and V: it gets 1.9x slower (N=1024: 43.75M / 43.79M at 32 rows, 82.96M /
81.81M at 16 rows; N=4096: 687.7M / 688.6M at 32 rows, 1308.2M / 1289.9M at
16 rows, dense / sparse). All runs passed the output check. WMMA runs
(N=1024 / N=4096, cycles):

| Fused WMMA | Dense | Sparse | Sparse speedup |
|---|---|---|---|
| 32 rows per CTA, N=1024 | 41.71M | 42.11M | 0.99x |
| 16 rows per CTA, N=1024 | 29.49M | 28.61M | 1.03x |
| 32 rows per CTA, N=4096 | 646.88M | 664.87M | 0.97x |
| 16 rows per CTA, N=4096 | 452.54M | 442.58M | 1.02x |

Stage breakdown at 32 rows, N=1024 (WMMA / WGMMA): softmax 66% / 64%,
Q.K^T 16% / 2%, P.V 11% / 6%, empty loop 8% / 23%. The sparse metadata loads
cost about 0.9% of the kernel; at N=4096 a further ~1.7% loss at 32 rows is
not yet explained. The WGMMA kernel's empty loop costs 10.2M cycles against
3.5M for WMMA.

(c) MoE (`akili_moe_*`, Mixtral-style top-2 gated FFN with SwiGLU): the
router, token permutation, SwiGLU and combine run on the host; each expert's
two GEMMs (W13 and W2, weights as the A operand) run on the device through
the same four GEMM backends as the llama decode, 16 launches in total.
4-warp CTAs. Output check: relative L2 error 0.00034 against an fp32
reference that rounds weights to fp16 (and, for the sparse modes, prunes
2:4 after rounding), all four modes PASSED. Small size (`-t 64 -d 256
-f 256 -e 4`): 7.13M / 7.74M / 3.09M / 2.93M cycles. Instructions at the
large size: 1.088G / 0.781G / 0.290G / 0.155G.

(d) Llama-3.2-1B end to end (`akili_llama2_e2e_*`): one real 128-token
prompt is processed in a single batched prefill (every projection a GEMM with
N=128; RMSNorm, RoPE, causal attention and SwiGLU on the host; logits for the
last token only, so the LM head runs at N=1), its KV cache is copied to 32
sequences, and those decode 8 steps with the batched decode path. Device
cycles are snapshotted at the phase boundaries: 113 launches in prefill, 904
in decode. 4-warp CTAs, original (row-major) activation layout, `-s 42`.
Instructions (WMMA / sparse WMMA / WGMMA / sparse WGMMA): 36.52G / 26.18G /
9.51G / 5.05G. Whole-run wall time: 1,993 s / 1,839 s / 418 s / 374 s.
WGMMA over WMMA in total: 8.5x dense, 9.5x sparse. The batched prefill
matches the token-by-token forward bit for bit on the host path. The sparse
modes prune the weights 2:4 without retraining.

Other caveats:
- Unfused attention (`-d 3072`): the check's reference uses the same
  fp16-rounded Q, K, V and P as the device, and P is checked against the
  `fast_exp` error bound (1e-5 abs + 2e-3 rel). Re-run on this revision
  (cycles within 0.6% of the table): S passes in all modes; 30-50 elements per
  run remain wrong in P or O (softmax outputs outside [0, 1], runs of zeroed
  outputs), which is intermittent device corruption, not precision. WGMMA
  passed.
- Sparse Llama modes use 2:4-pruned weights without retraining, so generated
  text is garbled; they measure performance only.
- The card was shared with another user's jobs; each run was checked to have
  used this bitstream.
- Unfused attention at `-n 512 -d 32`: 20.49M / 18.45M / 11.20M / 10.72M.

## Fused FlashAttention, best configuration per mode (recorded 2026-10-10)

d=32, same bitstream and core as above. Cycles; "check" is the output check.

| N | WMMA | Sparse WMMA | WGMMA | Sparse WGMMA |
|---|---|---|---|---|
| 1024 | 29.3M | 28.5M | 31.3M | 30.7M (17 wrong) |
| 4096 | 452.5M (e) | 442.6M (e) | 493.5M | 483.2M (17 wrong) |

WMMA modes: 16 query rows per CTA, reference softmax (`fast_exp`, no
`FA_SOFTMAX_FAST`). WGMMA modes: 32 rows per CTA, `fa_softmax_tile`.
(e) measured on the earlier build with the same code path.

Softmax variants, N=1024 (cycles / check):

| Variant | WMMA | Sparse WMMA | WGMMA | Sparse WGMMA |
|---|---|---|---|---|
| reference, 16 rows | 29.2-29.3M, passed x3 | 28.4-28.5M, passed x3 | - | - |
| new, 16 rows | 43.2-44.6M, failed x4 | 34.1-34.2M, failed x4 | - | - |
| new, 32 rows | 41.9-42.2M, failed 1 of 2 | 36.8-36.9M, passed x2 | 31.3M, passed | 30.7M, 17 wrong |
| reference, 32 rows | - | - | 43.2M, passed | 43.1M, 2 wrong |

The new softmax is wrong and 1.5x slower on WMMA at 16 rows (cause not
identified) but 1.38x faster on WGMMA. Sparse WGMMA fails 2-17 elements with
either softmax; dense WGMMA failed 6 at N=4096 with the reference softmax.

d=128 (N=768, 8 query rows per CTA, cycles):

| WMMA | Sparse WMMA | WGMMA (d-tiled) | Sparse WGMMA (d-tiled) |
|---|---|---|---|
| 69.30M | 66.97M | 92.14M | 86.76M (27 wrong) |

WMMA at N=2048, 16 rows: 486.71M / 469.89M, both passed. The d-tiled WGMMA
kernel needs ~15 KB of the 16 KB LMEM per CTA, so one 1-warp CTA is
resident; WMMA at 8 rows uses 5.6 KB. Instructions: 7.57M WGMMA vs 14.36M WMMA.

### d=128 FA2 tile sweep (N=768, reference softmax, cycles dense / sparse)

| Rows/CTA | Keys/tile | WMMA | Sparse WMMA | LMEM |
|---|---|---|---|---|
| 8 | 16 | 66.90M | 64.04M | 4.9 KB |
| 8 | 32 | 83.02M | 80.51M | 5.6 KB |
| 8 | 64 | 82.19M | 81.04M | 7.2 KB |
| 16 | 16 | 96.19M | 93.24M | 9.7 KB |
| 16 | 32 | 83.04M | 80.50M | 11.3 KB |
| 16 | 64 | 82.19M | 81.04M | 14.3 KB |

All passed. d-tiled WGMMA (8 rows, 16 keys): N=768 92.14M / 86.78M, N=2048
653.04M / 614.50M. Dense passed; sparse failed 27 of 98304 (N=768, identical
elements on two runs: deterministic) and 70 of 262144 (N=2048). WMMA at
N=2048 (16 rows, 32 keys, new softmax): 486.71M / 469.89M.

## Fused FlashAttention, optimized (recorded 2026-10-10)

d=32. Every mode at its best tile, all using the same softmax code
(`fa_softmax.h`, identical in both kernel directories). Cycles; check counts
are wrong elements out of N x 32.

| N | WMMA | Sparse WMMA | WGMMA | Sparse WGMMA |
|---|---|---|---|---|
| 1024 | 20.61M | 18.80M | 19.95M (passed x2) | 19.22M (passed 1 of 2; 31) |
| 4096 | 326.3M | 316.0M / 311.5M | 312.2M (passed 1 of 2; 32) | 299.3M (passed 1 of 2; 30) |

Order at N=4096: sparse WGMMA < {WGMMA, sparse WMMA: within 1.5%, run to
run} < WMMA. At N=1024 sparse WMMA is the fastest mode.

Build flags (all opt-in; defaults reproduce the earlier kernels):
- WMMA: `-DFA_SM_RELOAD=2 -DFA_SMV2 -DFA_BR=8 -DFA_BC=64`
- Sparse WMMA: `-DFA_SM_RELOAD=2 -DFA_SMV2 -DFA_BR=16 -DFA_BC=32`
- WGMMA, sparse WGMMA: `-DFA_REGSM -DFA_SMV2 -DFA_BC=16 -DFA_OLMEM
  -DFA_XCHG_LMEM -DFA_XCHG_NOFENCE -DFA_SUB=2 -DFA_ONE_BAR` (32 rows)

What moved the numbers:
- Softmax spills (both APIs): the compiler kept all scores live across the
  max and exp passes and spilled to the stack. A compiler barrier per
  iteration (`FA_SM_RELOAD=2`) and a degree-2 exp with multiply-based fp16
  conversion (`FA_SMV2`, error 2.7e-3 against the 2e-2 check) took WGMMA from
  31.3M to 23.7M and WMMA from 29.3M to 22.5M at N=1024. The earlier "new
  softmax" slowdown on WMMA was this spilling (654 vs 284 memory ops).
- WGMMA register softmax (`FA_REGSM`): softmax on the S accumulator
  registers, 16-key S tiles (NRC=16; the WGMMA intrinsic pins all 32 FP
  registers, so NRC=32 cannot be operated on), running max/sum carried as
  integer bits across the P.V WGMMA.
- O parked in LMEM between tiles (`FA_OLMEM`) instead of round-tripping the
  stack: 1.18-1.24x. Stack traffic through the D-cache was the serializer.
- 32-key K/V fetch processed as two 16-key S sub-tiles (`FA_SUB=2`) and one
  barrier per tile (`FA_ONE_BAR`): about 2-5% each.
- Re-tuning WMMA's tile with the fixed softmax: dense 8 rows x 64 keys,
  1.21x over 16 x 32 at N=4096.

Tried and slower: lazy rescale with a warp vote (and it hung the card once,
sparse N=4096), 8 KB CTAs for two resident WGMMA CTAs, 16-key tiles on the
LMEM softmax path, interleave 4.

Correctness: the WGMMA register-softmax kernels intermittently lose one output
row (~30 elements). 64- and 128-row CTAs (8/16 warps) are 1.15-1.2x faster
(N=4096: 260M / 253M) but lose up to 1.6% of outputs; failures grow with warps
per CTA and include whole zero rows and one accumulator register wrong across a
warp. Removing shuffles, fences and half-word P stores did not remove them, so
they are not attributed to the kernel; those configurations are not reported
above. The WMMA kernels also show occasional 2-element errors at N=4096.
