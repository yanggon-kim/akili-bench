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

## Results (device cycles)

Speedups are sparse over dense within the same API (above 1 = sparse faster).

| Workload | Problem size | WMMA | Sparse WMMA | WGMMA | Sparse WGMMA | sp/dense WMMA | sp/dense WGMMA |
|---|---|---|---|---|---|---|---|
| acccnn | `-c 128 -o 128 -h 64 -w 64 -s 3` | 140.42M | 125.78M | 44.82M | 34.53M | 1.12x | 1.30x |
| attention (unfused) | `-n 512 -d 3072` | 777.09M | 457.59M | 137.95M | 112.68M | 1.70x | 1.22x |
| Llama-3.2-1B prefill (a) | `-L 128` | 129.39G | 129.52G | 12.03G | 9.88G | 1.00x | 1.22x |
| Llama-3.2-1B decode | batch 32, `-n 8 -s 42` | 184.26G | 181.58G | 23.82G | 21.55G | 1.02x | 1.11x |
| Llama-3.2-1B decode, overall (b) | batch 32, `-n 8 -s 42` | 213.9G | 193.6G | 52.6G | 32.8G | 1.10x | 1.60x |
| NeRF, B tiled (c) | `-r 512 -s 64` | 1.215G | 1.160G | n/a | n/a | 1.05x | n/a |
| fused FlashAttention (d) | d=32, N=4096 | 647.52M | 664.07M | 688.08M | 689.95M | 0.98x | 1.00x |

## Notes

(a) Prefill: only WMMA was rerun on the fixed build (129.39G). Sparse WMMA,
WGMMA and sparse WGMMA are from the build before the `+xvortex` fix (old
build). Against the old dense WMMA run (132.46G), sparse WMMA is 1.02x.

(b) Decode overall: whole-program wall time at 160.8 MHz, i.e. device matmuls
plus host compute, weight preparation and PCIe transfers. Wall times:
WMMA 1,330 s, sparse WMMA 1,204 s, WGMMA 327 s, sparse WGMMA 204 s. The dense
hosts re-pack and re-upload the weights on every matmul; the sparse hosts
cache them on the device, so the overall comparison includes that host
difference.

(c) NeRF tiled-B: B is pre-tiled on the host for the two WMMA kernels
(`-DAKILI_NERF_TILEDB`), old build. WGMMA stages B through DXA, so there is
no tiled WGMMA run. Untiled, fixed build: WMMA 4.389G, sparse WMMA 10.186G,
WGMMA 0.935G, sparse WGMMA 0.905G (1.03x). All NeRF modes fail the output
check; timing only.

(d) Fused FlashAttention (online softmax, only Q.K^T sparse), d=32. WMMA at
N=4096 failed its output check (single bad row, intermittent); the other
three passed. Other sizes (WMMA / sparse WMMA / WGMMA / sparse WGMMA):

| N | WMMA | Sparse WMMA | WGMMA | Sparse WGMMA |
|---|---|---|---|---|
| 512 | 10.73M | 10.61M | 11.12M | 11.13M |
| 1024 | 41.86M (fail) | 42.08M | 43.74M | 43.81M |
| 2048 | 164.18M | 166.96M | 173.50M | 173.66M |

Other caveats:
- All four attention `-d 3072` runs fail the output check (the softmax stage
  is nondeterministic on the FPGA); cycles are timing only.
- Sparse decode and prefill modes use 2:4-pruned weights without retraining,
  so generated text is garbled; they measure performance only.
- The card was shared with another user's jobs; each run was checked to have
  used this bitstream.
- Unfused attention at `-n 512 -d 32`: 20.49M / 18.45M / 11.20M / 10.72M.
