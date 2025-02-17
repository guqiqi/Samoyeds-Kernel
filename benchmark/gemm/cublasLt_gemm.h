//
// Created by kiki on 24-1-18.
//

#ifndef FLEXIBLE_SPMM_CUBLASLT_GEMM_H
#define FLEXIBLE_SPMM_CUBLASLT_GEMM_H

#include <cuda_fp16.h>

void GemmLtExec(const int m, const int n, const int k,
              const half *A, const half *B, half *C);
#endif //FLEXIBLE_SPMM_CUBLASLT_GEMM_H
