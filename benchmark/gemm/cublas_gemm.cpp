//
// Created by kiki on 24-1-18.
//
#include <iostream>
#include <cublas_v2.h>

#include "cublas_gemm.h"

inline void checkCudaStatus(cudaError_t status) {
    if (status != cudaSuccess) {
        printf("cuda API failed with status %d: %s\n", status, cudaGetErrorString(status));
        throw std::logic_error("cuda API failed");
    }
}

inline void checkCublasStatus(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS API failed with status %d\n", status);
        throw std::logic_error("cuBLAS API failed");
    }
}

void GemmExec(const int m, const int n, const int k,
              const half *A, const half *B, half *C){
    const half alpha = 1.0;
    const half beta = 1.0;
    cublasHandle_t handle;
    cublasCreate(&handle);
    //    cublas进行转置计算
    // checkCublasStatus(cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, &alpha, A, k, B, n, &beta, C, m));
    // checkCublasStatus(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B, n, A, k, &beta, C, n));
    checkCublasStatus(cublasGemmEx(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, m, k,
            &alpha,
            B, CUDA_R_16F, n,
            A, CUDA_R_16F, k,
            &beta,
            C, CUDA_R_16F, n,
            CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
        )
    );

    cudaDeviceSynchronize();
}