// helper_vortex.h — Shared Vortex error checking and device init helpers
//
// Inspired by NVIDIA cuda-samples Common/helper_cuda.h
// Provides VX_CHECK() macro for all Vortex API calls.

#pragma once

#include <vortex.h>
#include <cstdio>
#include <cstdlib>

// Error-checking macro for Vortex API calls (analogous to checkCudaErrors)
#define VX_CHECK(expr)                                                  \
    do {                                                                \
        int _ret = (expr);                                              \
        if (_ret != 0) {                                                \
            fprintf(stderr, "Vortex Error: '%s' returned %d at %s:%d\n", \
                    #expr, _ret, __FILE__, __LINE__);                   \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

#ifdef DEBUG
    #define VX_DBG(x) x
#else
    #define VX_DBG(x)
#endif
