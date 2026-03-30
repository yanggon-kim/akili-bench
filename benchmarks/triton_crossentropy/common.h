#ifndef _COMMON_H_
#define _COMMON_H_

typedef struct {
  uint32_t num_rows;     // batch size
  uint32_t num_classes;  // vocabulary size
  uint64_t logits_addr;  // input logits [num_rows × num_classes]
  uint64_t labels_addr;  // target labels [num_rows] (int32)
  uint64_t loss_addr;    // output per-row loss [num_rows]
} kernel_arg_t;

#endif
