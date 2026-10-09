#ifndef AKILI_ABLOCK_H
#define AKILI_ABLOCK_H

// Block-tiled A layout shared by the WMMA and WGMMA GEMM paths under
// AKILI_A_BLOCKED: [rows/8][cols/32][8][32], row-major inside each block, so a
// tile never strides by a power-of-two row pitch. cols counts stored elements
// (compressed columns for 2:4 sparse A). Stacking consecutive row-blocks of one
// column-block gives the same [rows][32] row-major tile a DXA stage expects.

#include <stdint.h>

constexpr uint32_t kABlockRows = 8;
constexpr uint32_t kABlockCols = 32;
constexpr uint32_t kABlockElems = kABlockRows * kABlockCols;

#ifndef __riscv
#include <cstring>
#include <vector>

template <typename T>
inline void akili_block_a(std::vector<T>& a, uint32_t rows, uint32_t cols) {
  std::vector<T> src(a);
  uint32_t col_blocks = cols / kABlockCols;
  for (uint32_t r = 0; r < rows; ++r)
    for (uint32_t cb = 0; cb < col_blocks; ++cb)
      std::memcpy(&a[((size_t)(r / kABlockRows) * col_blocks + cb) * kABlockElems
                     + (r % kABlockRows) * kABlockCols],
                  &src[(size_t)r * cols + cb * kABlockCols],
                  kABlockCols * sizeof(T));
}
#endif

#endif
