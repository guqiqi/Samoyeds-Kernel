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

#include <iostream>
#include <cublasLt.h>

#include "cublasLt_gemm.h"

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

void GemmLtExec(const int m, const int n, const int k,
              const half *A, const half *B, half *C){
    const half alpha = 1.0;
    const half beta = 0.0;

    cublasLtHandle_t ltHandle;
    size_t workspaceSize = 1024 * 1024 * 4;
    void *workspace;

    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    cublasLtMatmulPreference_t preference = NULL;

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;

    int returnedResults                             = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};

    checkCublasStatus(cublasLtCreate(&ltHandle));
    checkCudaStatus(cudaMalloc(( void ** )&workspace, workspaceSize));

    int lda = m, ldb=k, ldc=m;

    // create operation desciriptor; see cublasLtMatmulDescAttributes_t for details about defaults; here we just need to
    // set the transforms for A and B
    checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    // create matrix descriptors, we are good with the details here so no need to set any extra attributes
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, m, n, ldc));

    // create preference handle; here we could use extra attributes to disable tensor ops or to make sure algo selected
    // will work with badly aligned A, B, C; here for simplicity we just assume A,B,C are always well aligned (e.g.
    // directly come from cudaMalloc)
    checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
    checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

    // we just need the best available heuristic to try and run matmul. There is no guarantee this will work, e.g. if A
    // is badly aligned, you can request more (e.g. 32) algos and try to run them one by one until something works
    checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));

    if (returnedResults == 0) {
        checkCublasStatus(CUBLAS_STATUS_NOT_SUPPORTED);
    }

    checkCublasStatus(cublasLtMatmul(ltHandle,
                                     operationDesc,
                                     &alpha,
                                     A,
                                     Adesc,
                                     B,
                                     Bdesc,
                                     &beta,
                                     C,
                                     Cdesc,
                                     C,
                                     Cdesc,
                                     &heuristicResult.algo,
                                     workspace,
                                     workspaceSize,
                                     0));

    // descriptors are no longer needed as all GPU work was already enqueued
    if (preference) checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
    if (Cdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
    if (operationDesc) checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));
////    auto lda = (transa == CUBLAS_OP_N) ? k : m;
////    //auto     lda            = (!is_rowmajor) ? num_A_cols : num_A_rows;
////    auto ldb = (transa == CUBLAS_OP_N) ? n : k;
////    //auto     ldb            = (!is_rowmajor) ? num_B_cols : num_B_rows;
////    auto ldc = (transa == CUBLAS_OP_N) ? n : m;
//
//    int lda = m, ldb=k, ldc = m;
//
//    void *workspace;
//    size_t workspaceSize = 1024 * 1024 * 8;
//    int returnedResults                             = 0;
//    cublasLtMatmulHeuristicResult_t heuristicResult = {};
//
//    checkCublasStatus(cublasLtCreate(&ltHandle));
//    checkCudaStatus(cudaMalloc(( void ** )&workspace, workspaceSize));
//
//// cublasLt GEMM描述符
//// TODO 确定数据类型
////    checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_16F, CUDA_R_16F));
//    checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
////
////    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &CUBLAS_OP_N, sizeof(CUBLAS_OP_N));
////    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &CUBLAS_OP_N, sizeof(CUBLAS_OP_N));
//    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
//    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
//
////    checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16F, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
////    checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16F, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
////    checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16F, m, n, ldc));
//    checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
//    checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
//    checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, m, n, ldc));
//
//    // Create preference handle; In general, extra attributes can be
//    // used here to disable tensor ops or to make sure algo selected
//    // will work with badly aligned A, B, C. However, for simplicity
//    // here we assume A,B,C are always well aligned (e.g., directly
//    // come from cudaMalloc)
//    checkCublasStatus( cublasLtMatmulPreferenceCreate( &preference ) );
//    checkCublasStatus( cublasLtMatmulPreferenceSetAttribute(
//            preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof( workspaceSize ) ) );
//
//    //  cublasLt GEMM算法推荐
//    // We just need the best available heuristic to try and run matmul.
//    // There is no guarantee that this will work. For example, if A is
//    // badly aligned, you can request more (e.g. 32) algos and try to
//    // run them one by one until something works.
//    checkCublasStatus( cublasLtMatmulAlgoGetHeuristic(
//            ltHandle,
//            operationDesc,
//            Adesc,
//            Bdesc,
//            Cdesc,
//            Cdesc,
//            preference,
//            1,
//            &heuristicResult,
//            &returnedResults ) );
//    if (returnedResults == 0) {
//        checkCublasStatus(CUBLAS_STATUS_NOT_SUPPORTED);
//    }
//
//    int alpha = 1, beta = 0;
//    checkCublasStatus(cublasLtMatmul(ltHandle,
//                                     operationDesc,
//                                     &alpha,
//                                     A,
//                                     Adesc,
//                                     B,
//                                     Bdesc,
//                                     &beta,
//                                     C,
//                                     Cdesc,
//                                     (void *) C,
//                                     Cdesc,
//                                     &heuristicResult.algo,
//                                     workspace,
//                                     workspaceSize,
//                                     nullptr));

    cudaDeviceSynchronize();
}