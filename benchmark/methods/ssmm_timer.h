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


#ifndef FLEXIBLE_SPMM_SSMM_TIMER_H
#define FLEXIBLE_SPMM_SSMM_TIMER_H

#define WARMUP 10
#define TIMES 1000

#include <iostream>
#include "../format/formats.hpp"
#include "../ssmm/horizontal_ssmm_kernel_op.h"
#include "../util/matrix_utils.h"

template<typename T>
void SSMM_timer(int m, int k, int n, int N, int M, int vector_length, uint seed) {
    auto *format = new Format_ss<T>(m, k, seed, vector_length, N, M);
    format->init();
    format->sync_device();

    int dense_B_lens = n / 4;
    HorizontalSparseMatrix<T> B;
    ZerosMatrix<T> C;
    // HorizontalSparseMatrix是转置的矩阵, 所以行列数是相反的
    B.init_DataMatrix(n, k, dense_B_lens, true);
    C.init_DataMatrix(m, dense_B_lens);

    using BlockShape = ShapeBase<128, 32, 64>;
    using WarpShape = ShapeBase<32, 32, 64>;
    using MmaShape = ShapeBase<16, 32, 8>;
    const int stage = 2;

    float time_ms;
    cudaEvent_t start, stop;
    cudaDeviceSynchronize();
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < WARMUP; i++) {
        HorizontalSsmmKernelExec<BlockShape, WarpShape, MmaShape, stage>(
                m, n, k, vector_length, N, M,
                format->devicePtr._value,
                format->devicePtr._meta,
                format->devicePtr._indices,
                B.device_ptr, B.indices_device_ptr, B.indices_len, C.device_ptr, C.device_ptr);
    }
    cudaEventRecord(start, 0);
    for (int i = 0; i < TIMES; i++) {
        HorizontalSsmmKernelExec<BlockShape, WarpShape, MmaShape, stage>(
                m, n, k, vector_length, N, M,
                format->devicePtr._value,
                format->devicePtr._meta,
                format->devicePtr._indices,
                B.device_ptr, B.indices_device_ptr, B.indices_len, C.device_ptr, C.device_ptr);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    time_ms = time_ms / (float)TIMES;
    // Print the duration
    // std::cout << "One iter for SSMM kernel is :" << time_ms << " ms\n";

    std::cout << time_ms << std::endl;



}
#endif //FLEXIBLE_SPMM_SSMM_TIMER_H
