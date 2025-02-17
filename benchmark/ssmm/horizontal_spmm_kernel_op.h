//
// Created by cpwu on 12/29/23.
//

#ifndef FLEXIBLE_SPMM_HORIZONTAL_SPMM_KERNEL_OP_H
#define FLEXIBLE_SPMM_HORIZONTAL_SPMM_KERNEL_OP_H

#include "horizontal_spmm_kernel.h"
#include "../util/utils.h"

template<typename KernelType>
__global__ void _HorizontalSpmmKernel(const int m, const int n, const int k,
                                      const half *A_values, const uint *A_metadata, const uint *A_indices,
                                      const half *B, const half *C, half *D) {
    KernelType kernel;
    extern __shared__ half shared_mem_workspace[];
    kernel.mainLoop(m, n, k, A_values, A_metadata, A_indices, B, shared_mem_workspace);
    kernel.epilogue(m, n, D, shared_mem_workspace);
}

template<typename KernelType>
__global__ void _HorizontalSpmmTransKernel(const int m, const int n, const int k,
                                          const half *A_values, const uint *A_metadata, const uint *A_indices,
                                          const half *B, const half *C, half *D) {
    KernelType kernel;
    extern __shared__ half shared_mem_workspace[];
    kernel.mainLoopTrans(m, n, k, A_values, A_metadata, A_indices, B, shared_mem_workspace);
    kernel.epilogueDenseTrans(m, n, D, shared_mem_workspace);
}

template<typename KernelType>
__global__ void _HorizontalSpmmSparseWeightedTransKernel(const int m, const int n, const int k,
                                                         const half *A_values, const uint *A_metadata, const uint *A_indices,
                                                         const half *B, const uint *B_indices, const uint B_indices_len,
                                                         const half *routing_weights, const half *C, half *D) {
    KernelType kernel;
    extern __shared__ half shared_mem_workspace[];
    kernel.mainLoopTransWeightDot(m, n, k, A_values, A_metadata, A_indices, B, routing_weights,
                    shared_mem_workspace);
    kernel.epilogueSparseTrans(m, n, B_indices, B_indices_len, routing_weights, D, shared_mem_workspace);
}

template<typename KernelType>
__global__ void _HorizontalSpmmDenseWeightedTransKernel(const int m, const int n, const int k,
                                                 const half *A_values, const uint *A_metadata, const uint *A_indices,
                                                 const half *B,
                                                 const half *routing_weights, const half *C, half *D) {
    KernelType kernel;
    extern __shared__ half shared_mem_workspace[];
    kernel.mainLoopTransWeightDot(m, n, k, A_values, A_metadata, A_indices, B, routing_weights,
                               shared_mem_workspace);
    kernel.epilogueDenseTrans(m, n, D, shared_mem_workspace);
}

template<typename BlockShape, typename WarpShape, typename MmaShape,
        int NStage>
void HorizontalSpmmKernelExec(const int m, const int n, const int k,
                              const int V, const int N, const int M,
                              const half *A_values, const uint *A_metadata, const uint *A_indices,
                              const half *B, const half *C, half *D) {
    using AccumulatorType = half;
    using ASwizzle = Swizzle8BWiseXor;
    using BSwizzle = Swizzle8BWiseXor;
    using CSwizzle = Swizzle8BWiseXor;

    using KernelType = HorizontalSpmmKernel<SparseRatioBase<1, 2>, 128, BlockShape, WarpShape, MmaShape, NStage,
                AccumulatorType, ASwizzle, BSwizzle, CSwizzle>;
    size_t shared_mem_size = max(KernelType::input_buffer_size, KernelType::output_buffer_size);
    cudaStream_t stream = NULL;
    int dim_x = m / KernelType::Block_M / M * N;
    int dim_y = n / KernelType::Block_N;
    _HorizontalSpmmKernel<KernelType><<<dim3(dim_x, dim_y, 1), dim3(128, 1, 1), shared_mem_size, stream>>>(
            m, n, k, A_values, A_metadata, A_indices, B, C, D);
}

template<typename BlockShape, typename WarpShape, typename MmaShape,
        int NStage>
void HorizontalSpmmTransKernelExec(const int m, const int n, const int k,
                              const int V, const int N, const int M,
                              const half *A_values, const uint *A_metadata, const uint *A_indices,
                              const half *B, const half *C, half *D) {
    using AccumulatorType = half;
    using ASwizzle = Swizzle8BWiseXor;
    using BSwizzle = Swizzle8BWiseXor;
    using CSwizzle = Swizzle8BWiseXor;

    using KernelType = HorizontalSpmmKernel<SparseRatioBase<1, 2>, 128, BlockShape, WarpShape, MmaShape, NStage,
            AccumulatorType, ASwizzle, BSwizzle, CSwizzle>;
    size_t shared_mem_size = max(KernelType::input_buffer_size, KernelType::output_buffer_size);
    cudaStream_t stream = NULL;
    int dim_x = m / KernelType::Block_M / M * N;
    int dim_y = n / KernelType::Block_N;
    _HorizontalSpmmTransKernel<KernelType><<<dim3(dim_x, dim_y, 1), dim3(128, 1, 1), shared_mem_size, stream>>>(
            m, n, k, A_values, A_metadata, A_indices, B, C, D);
}

template<typename BlockShape, typename WarpShape, typename MmaShape,
        int NStage>
void HorizontalSpmmSparseWeightedTransKernelExec(const int m, const int n, const int k,
                              const int V, const int N, const int M,
                              const half *A_values, const uint *A_metadata, const uint *A_indices,
                              const half *B, const uint *B_indices, const uint B_indices_len,
                              const half *routing_weights,
                              const half *C, half *D) {
    using AccumulatorType = half;
    using ASwizzle = Swizzle8BWiseXor;
    using BSwizzle = Swizzle8BWiseXor;
    using CSwizzle = Swizzle8BWiseXor;

    using KernelType = HorizontalSpmmKernel<SparseRatioBase<1, 2>, 128, BlockShape, WarpShape, MmaShape, NStage,
            AccumulatorType, ASwizzle, BSwizzle, CSwizzle>;
    size_t shared_mem_size = max(KernelType::input_buffer_size, KernelType::output_buffer_size);
    cudaStream_t stream = NULL;
    int dim_x = m / KernelType::Block_M / M * N;
    int dim_y = n / KernelType::Block_N;
    _HorizontalSpmmSparseWeightedTransKernel<KernelType><<<dim3(dim_x, dim_y, 1), dim3(128, 1, 1), shared_mem_size, stream>>>(
            m, n, k, A_values, A_metadata, A_indices, B, B_indices, B_indices_len, routing_weights, C, D);
}

template<typename BlockShape, typename WarpShape, typename MmaShape,
        int NStage>
void HorizontalSpmmDenseWeightedTransKernelExec(const int m, const int n, const int k,
                                                 const int V, const int N, const int M,
                                                 const half *A_values, const uint *A_metadata, const uint *A_indices,
                                                 const half *B,
                                                 const half *routing_weights,
                                                 const half *C, half *D) {
    using AccumulatorType = half;
    using ASwizzle = Swizzle8BWiseXor;
    using BSwizzle = Swizzle8BWiseXor;
    using CSwizzle = Swizzle8BWiseXor;

    using KernelType = HorizontalSpmmKernel<SparseRatioBase<1, 2>, 128, BlockShape, WarpShape, MmaShape, NStage,
            AccumulatorType, ASwizzle, BSwizzle, CSwizzle>;
    size_t shared_mem_size = max(KernelType::input_buffer_size, KernelType::output_buffer_size);
    cudaStream_t stream = NULL;
    int dim_x = m / KernelType::Block_M / M * N;
    int dim_y = n / KernelType::Block_N;
    _HorizontalSpmmDenseWeightedTransKernel<KernelType><<<dim3(dim_x, dim_y, 1), dim3(128, 1, 1), shared_mem_size, stream>>>(
            m, n, k, A_values, A_metadata, A_indices, B, routing_weights, C, D);
}


template void HorizontalSpmmKernelExec<
        ShapeBase<64, 32, 64>,
        ShapeBase<16, 32, 64>,
        ShapeBase<16, 32, 8>, 2>(
        int m, int n, const int k,
        const int V, const int N, const int M,
        const half *A_values, const uint *A_metadata, const uint *A_indices,
        const half *routing_weights,
        const half *C, half *D);

template void HorizontalSpmmTransKernelExec<
        ShapeBase<64, 32, 64>,
        ShapeBase<16, 32, 64>,
        ShapeBase<16, 32, 8>, 2>(
        int m, int n, const int k,
        const int V, const int N, const int M,
        const half *A_values, const uint *A_metadata, const uint *A_indices,
        const half *routing_weights,
        const half *C, half *D);

template void HorizontalSpmmSparseWeightedTransKernelExec<
        ShapeBase<64, 32, 64>,
        ShapeBase<16, 32, 64>,
        ShapeBase<16, 32, 8>, 2>(
        int m, int n, const int k,
        const int V, const int N, const int M,
        const half *A_values, const uint *A_metadata, const uint *A_indices,
        const half *B, const uint *B_indices, const uint B_indices_len,
        const half *routing_weights,
        const half *C, half *D);

template void HorizontalSpmmDenseWeightedTransKernelExec<
        ShapeBase<64, 32, 64>,
        ShapeBase<16, 32, 64>,
        ShapeBase<16, 32, 8>, 2>(const int m, const int n, const int k,
        const int V, const int N, const int M,
        const half *A_values, const uint *A_metadata, const uint *A_indices,
        const half *B,
        const half *routing_weights,
        const half *C, half *D);
#endif //FLEXIBLE_SPMM_HORIZONTAL_SPMM_KERNEL_OP_H
