/*
 * Copyright (c) 2025 Chenpeng Wu (cpwu_sjtu@sjtu.edu.cn), Qiqi Gu (qiqi.gu@sjtu.edu.cn). 
 * Copyright (C) 2023 Roberto Lopez Castro (roberto.lopez.castro@udc.es). All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef FLEXIBLE_SPMM_MEMCPY_PIPELINE_H
#define FLEXIBLE_SPMM_MEMCPY_PIPELINE_H

template<int NStage, bool UseMinSync>
struct Pipeline;

template<int NStage>
struct Pipeline<NStage, false> {
    //static_assert(NStage>1);

    __device__ __forceinline__
    void acquire_writer() {
    }

    __device__ __forceinline__
    void commit_stage() {
        asm volatile("cp.async.commit_group;\n"::);
    }

    __device__ __forceinline__
    void acquire_reader() {
        asm volatile ("cp.async.wait_group %0;\n"::"n"(NStage - 1));
        __syncthreads();
    }

    __device__ __forceinline__
    void release_reader() {
        __syncthreads();
    }
};

template<int NStage>
struct Pipeline<NStage, true> {
    //static_assert(NStage>1);
    int ahead_stage = 0;

    __device__ __forceinline__
    void acquire_writer() {
        if (ahead_stage == NStage - 1) {
            asm volatile ("cp.async.wait_group %0;\n"::"n"(NStage - 2));
            __syncthreads();
        }
    }

    __device__ __forceinline__
    void commit_stage() {
        asm volatile("cp.async.commit_group;\n"::);
        ahead_stage++;
    }

    __device__ __forceinline__
    void acquire_reader() {
    }

    __device__ __forceinline__
    void release_reader() {
        ahead_stage--;
    }
};

#endif //FLEXIBLE_SPMM_MEMCPY_PIPELINE_H
