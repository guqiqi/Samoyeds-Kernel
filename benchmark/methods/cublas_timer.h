//
// Created by cpwu on 24-4-9.
//

#ifndef FLEXIBLE_SPMM_CUBLAS_TIMER_H
#define FLEXIBLE_SPMM_CUBLAS_TIMER_H

#include <iostream>
#include "../format/formats.hpp"
#include "../util/matrix_utils.h"

#include "../gemm/cublas_gemm.h"

template<typename T>
void cuBlas_timer(int m, int k, int n, int N, int M, int vector_length, uint seed) {
    auto *format = new Format_ss<T>(m, k, seed, vector_length, N, M);
    format->init();
    format->sync_device();

    RandomMatrix<T> B;
    ZerosMatrix<T> C;
    B.init_DataMatrix(k, n);
    C.init_DataMatrix(m, n);

    GemmExec(m, n, k, format->devicePtr._pruned_value, B.device_ptr, C.device_ptr);

    using BlockShape = ShapeBase<128, 32, 64>;
    using WarpShape = ShapeBase<32, 32, 64>;
    using MmaShape = ShapeBase<16, 32, 8>;

    // Warm Up
    for (int i = 0; i < 10; i++) {
        GemmExec(m, n, k, format->devicePtr._pruned_value, B.device_ptr, C.device_ptr);
    }
    // Get the start time
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10000; i++) {
        GemmExec(m, n, k, format->devicePtr._pruned_value, B.device_ptr, C.device_ptr);
    }
    // Get the end time
    auto end = std::chrono::high_resolution_clock::now();
    // Calculate the duration
    std::chrono::duration<double> diff = end - start;
    // Print the duration
    std::cout << "Time to execute SSMM kernels 10000 times: " << diff.count() << " s\n";
}
#endif //FLEXIBLE_SPMM_CUBLAS_TIMER_H
