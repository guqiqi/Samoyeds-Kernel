//
// Created by cpwu on 12/29/23.
//

#ifndef FLEXIBLE_SPMM_HORIZONTAL_SSMM_KERNEL_OP_H
#define FLEXIBLE_SPMM_HORIZONTAL_SSMM_KERNEL_OP_H

#include "horizontal_ssmm_kernel.h"
#include "../util/utils.h"

template<typename KernelType>
__global__ void _HorizontalSsmmKernel(const int m, const int n, const int k,
                                      // const int V, const int N, const int M,
                                      const half *A_values, const uint *A_metadata, const uint *A_indices,
                                      const half *B, const uint *B_indices, const uint B_indices_len,
                                      const half *C, half *D) {
    KernelType kernel;
    extern __shared__ half shared_mem_workspace[];

    kernel.mainLoop(m, n, k, 0, 0, A_values, A_metadata, A_indices, B, B_indices, B_indices_len,
                    shared_mem_workspace);
    kernel.epilogue(m, n, B_indices_len, C, D, shared_mem_workspace);
}

template<typename KernelType>
__global__ void _HorizontalSsmmTransKernel(const int m, const int n, const int k,
                                      const half *A_values, const uint *A_metadata, const uint *A_indices,
                                      const half *B, const uint *B_indices, const uint B_indices_len,
                                      const half *C, half *D) {
    KernelType kernel;
    extern __shared__ half shared_mem_workspace[];

    kernel.mainLoopTrans(m, n, k, 0, 0, A_values, A_metadata, A_indices, B, B_indices, B_indices_len,
                    shared_mem_workspace, false);
    kernel.epilogueTrans(m, n, B_indices_len, D, shared_mem_workspace);
}

template<typename KernelType>
__global__ void _HorizontalSsmmFusedActTransKernel(const int m, const int n, const int k,
                                           const half *A_values, const uint *A_metadata, const uint *A_indices,
                                           const half *B, const uint *B_indices, const uint B_indices_len,
                                           const half *C, half *D) {
    KernelType kernel;
    extern __shared__ half shared_mem_workspace[];

    kernel.mainLoopTrans(m, n, k, 0, 0, A_values, A_metadata, A_indices, B, B_indices, B_indices_len,
                         shared_mem_workspace, true);
    kernel.epilogueTrans(m, n, B_indices_len, D, shared_mem_workspace);
}

template<typename BlockShape, typename WarpShape, typename MmaShape,
        int NStage>
void HorizontalSsmmKernelExec(const int m, const int n, const int k,
                              const int V, const int N, const int M,
                              const half *A_values, const uint *A_metadata, const uint *A_indices,
                              const half *B, const uint *B_indices, const uint B_indices_len,
                              const half *C, half *D) {
    using AccumulatorType = half;
    using ASwizzle = Swizzle8BWiseXor;
    using BSwizzle = Swizzle8BWiseXor;
    using CSwizzle = Swizzle8BWiseXor;

    using KernelType = HorizontalSsmmKernel<SparseRatioBase<1, 2>, 128, BlockShape, WarpShape, MmaShape, NStage,
                AccumulatorType, ASwizzle, BSwizzle, CSwizzle>;
    size_t shared_mem_size = max(KernelType::input_buffer_size, KernelType::output_buffer_size);
    cudaStream_t stream = NULL;
    int dim_x = m / KernelType::Block_M / M * N;
    int dim_y = B_indices_len / KernelType::Block_N;
    _HorizontalSsmmKernel<KernelType><<<dim3(dim_x, dim_y, 1), dim3(128, 1, 1), shared_mem_size, stream>>>(
            m, n, k, A_values, A_metadata, A_indices, B, B_indices, B_indices_len, C, D);
}

template<typename BlockShape, typename WarpShape, typename MmaShape,
        int NStage>
void HorizontalSsmmTransKernelExec(const int m, const int n, const int k,
                              const int V, const int N, const int M,
                              const half *A_values, const uint *A_metadata, const uint *A_indices,
                              const half *B, const uint *B_indices, const uint B_indices_len,
                              const half *C, half *D) {
    using AccumulatorType = half;
    using ASwizzle = Swizzle8BWiseXor;
    using BSwizzle = Swizzle8BWiseXor;
    using CSwizzle = Swizzle8BWiseXor;

    using KernelType = HorizontalSsmmKernel<SparseRatioBase<1, 2>, 128, BlockShape, WarpShape, MmaShape, NStage,
            AccumulatorType, ASwizzle, BSwizzle, CSwizzle>;
    size_t shared_mem_size = max(KernelType::input_buffer_size, KernelType::output_buffer_size);
    cudaStream_t stream = NULL;
    int dim_x = m / KernelType::Block_M / M * N;
    int dim_y = B_indices_len / KernelType::Block_N;
    _HorizontalSsmmTransKernel<KernelType><<<dim3(dim_x, dim_y, 1), dim3(128, 1, 1), shared_mem_size, stream>>>(
            m, n, k, A_values, A_metadata, A_indices, B, B_indices, B_indices_len, C, D);
}

template<typename BlockShape, typename WarpShape, typename MmaShape,
        int NStage>
void HorizontalSsmmFusedActTransKernelExec(const int m, const int n, const int k,
                                   const int V, const int N, const int M,
                                   const half *A_values, const uint *A_metadata, const uint *A_indices,
                                   const half *B, const uint *B_indices, const uint B_indices_len,
                                   const half *C, half *D) {
    using AccumulatorType = half;
    using ASwizzle = Swizzle8BWiseXor;
    using BSwizzle = Swizzle8BWiseXor;
    using CSwizzle = Swizzle8BWiseXor;

    using KernelType = HorizontalSsmmKernel<SparseRatioBase<1, 2>, 128, BlockShape, WarpShape, MmaShape, NStage,
            AccumulatorType, ASwizzle, BSwizzle, CSwizzle>;
    size_t shared_mem_size = max(KernelType::input_buffer_size, KernelType::output_buffer_size);
    cudaStream_t stream = NULL;
    int dim_x = m / KernelType::Block_M / M * N;
    int dim_y = B_indices_len / KernelType::Block_N;
    _HorizontalSsmmFusedActTransKernel<KernelType><<<dim3(dim_x, dim_y, 1), dim3(128, 1, 1), shared_mem_size, stream>>>(
            m, n, k, A_values, A_metadata, A_indices, B, B_indices, B_indices_len, C, D);
}

#endif //FLEXIBLE_SPMM_HORIZONTAL_SSMM_KERNEL_OP_H
