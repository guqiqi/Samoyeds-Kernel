//
// Created by cpwu on 12/29/23.
//

#ifndef FLEXIBLE_SPMM_SPMM_UTILS_H
#define FLEXIBLE_SPMM_SPMM_UTILS_H

#include <cuda_fp16.h>
#include <stdio.h>

// *** type for storing a 3D shape  ***
template<int M_, int K_, int N_>
struct ShapeBase {
    static constexpr int M = M_, K = K_, N = N_;
};

template<int N_, int M_>
struct SparseRatioBase {
    static constexpr int N = N_, M = M_;
};

struct SwizzleIdentity {
    __device__ __forceinline__
    int operator()(int o) {
        return o;
    }
};

struct Swizzle8BWiseXor {
    __device__ __forceinline__
    int operator()(int o) {
        return (o ^ ((o & (7 << 6)) >> 3));
    }
};

#endif //FLEXIBLE_SPMM_SPMM_UTILS_H
