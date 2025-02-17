//
// Created by cpwu on 24-4-9.
//

#ifndef FLEXIBLE_SPMM_CUBLASLT_TIMER_H
#define FLEXIBLE_SPMM_CUBLASLT_TIMER_H

#include <iostream>
#include "../format/formats.hpp"
#include "../util/matrix_utils.h"

#include "../gemm/cublasLt_gemm.h"

template<typename T>
void cuBlasLt_timer(int m, int k, int n, int N, int M, int vector_length, uint seed) {
    auto *format = new Format_ss<T>(m, k, seed, vector_length, N, M);
    format->init();
    format->sync_device();

    int dense_B_lens = n / 4;
    RandomMatrix<T> B;
    ZerosMatrix<T> C;
    // HorizontalSparseMatrix是转置的矩阵, 所以行列数是相反的
    B.init_DataMatrix(k, n);
    C.init_DataMatrix(m, n);

    GemmExec(m, n, k, format->devicePtr._pruned_value, B.device_ptr, C.device_ptr);

    using BlockShape = ShapeBase<128, 32, 64>;
    using WarpShape = ShapeBase<32, 32, 64>;
    using MmaShape = ShapeBase<16, 32, 8>;

    // Warm Up
    for (int i = 0; i < 10; i++) {
        GemmLtExec(m, n, k, format->devicePtr._pruned_value, B.device_ptr, C.device_ptr);
    }
    // Get the start time
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10000; i++) {
        GemmLtExec(m, n, k, format->devicePtr._pruned_value, B.device_ptr, C.device_ptr);
    }
    // Get the end time
    auto end = std::chrono::high_resolution_clock::now();
    // Calculate the duration
    std::chrono::duration<double> diff = end - start;
    // Print the duration
    std::cout << "Time to execute SSMM kernels 10000 times: " << diff.count() << " s\n";
}

#endif //FLEXIBLE_SPMM_CUBLASLT_TIMER_H
