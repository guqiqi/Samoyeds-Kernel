/*
 * Copyright (c) 2025 Chenpeng Wu (cpwu_sjtu@sjtu.edu.cn), Qiqi Gu (qiqi.gu@sjtu.edu.cn). 
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
