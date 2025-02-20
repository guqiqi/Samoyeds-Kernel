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

#ifndef FLEXIBLE_SPMM_HORIZONTAL_SSMM_KERNEL_H
#define FLEXIBLE_SPMM_HORIZONTAL_SSMM_KERNEL_H

#include "data_structure_helper.h"
#include "copy_helper.h"
#include "mma_helper.h"
#include "../mm_utils/memcpy_pipeline.h"
// #include "spmm_debug_helper.h"

#define TOTAL_THREADS_PER_BLOCK 128

#define METADATA_SIZE 2  // 每个metadata占两个bit
#define BITS_PER_BYTE 8  // 1 Byte = 8 Bits
// OPTIMIZE: sizeof(uint)是否被编译器所优化？
#define NUM_OF_META_PER_UINT static_cast<int>(sizeof(uint) * BITS_PER_BYTE / METADATA_SIZE)
#define THREADS_PER_WARP 32 // 每个warp有32个线程
#define HALF_PER_UINT 2 // 每个uint可以存放2个half
#define SPTC_N 2 // SPTC (sparse tensor core) N:M
#define SPTC_M 4 // SPTC (sparse tensor core) M:M
#define A_INDICES_BUFFER_NUM 2

// Macro definitions for A_GM_TO_SM_COPY
#define A_GM_TO_SM_CP_SIZE 8 // 每个线程一次A_value拷贝GM->SM的大小，以half为单位
#define A_GM_TO_SM_CP_THREAD_PER_ROW 2 // 一次A_value拷贝GM->SM中，每两个线程负责1行的拷贝
#define A_GM_TO_SM_CP_ROWS_PER_ITER 64 // 一次A_value拷贝GM->SM中，每次拷贝64行

#define A_INDICES_GM_TO_SM_CP_COLS_PER_ITER 64 // 一次A_indices拷贝GM->SM中，每次拷贝64列

// Macro definitions for B_GM_TO_SM_COPY
#define B_GM_TO_SM_CP_SIZE 16 // 每个线程一次B_value拷贝GM->SM的大小，以half为单位
#define B_GM_TO_SM_CP_TIMES 2 // cp.async指令最大支持的拷贝大小为16B，因此需要分两次拷贝
#define B_GM_TO_SM_CP_THREAD_PER_ROW 2 // 一次B_value拷贝GM->SM中，每两个线程负责1行的拷贝
#define B_GM_TO_SM_CP_ROWS_PER_ITER 64 // 一次B_value拷贝GM->SM中，每次拷贝64行

// Macro definitions for C_GM_TO_FRAGMENT_COPY
#define C_FRAGMENT_TO_SM_CP_COLS_PER_ITER 128

// Macro definitions for C_SM_TO_GM_COPY
#define C_SM_TO_GM_CP_SIZE 64 // 每个线程一次C拷贝SM->GM的大小，以half为单位
#define C_SM_TO_GM_CP_TIMES 8 // 直接赋值操作支持float4(8half)类型的拷贝，因此需要分8次拷贝
#define C_SM_TO_GM_CP_ROWS_PER_ITER 128 // 一次C拷贝SM->GM中，每次拷贝128行
#define C_SM_TO_GM_CP_COLS_PER_ITER 64 // 一次C拷贝SM->GM中，每次拷贝64列

template<
        typename SparseRatio, int VectorLength,
        // tiling shapes
        typename BlockShape, typename WarpShape, typename MmaShape,
        // threadblock level pipeline stage
        int ThreadBlockStage,
        // type of accumulator
        typename AccumulatorType,
        // type of shared memory swizzling
        typename ASwizzle, typename BSwizzle, typename CSwizzle>
struct HorizontalSsmmKernel {
    static constexpr int N = SparseRatio::N;
    static constexpr int M = SparseRatio::M;
    static constexpr int vector_length = VectorLength;
    static constexpr int Block_M = BlockShape::M;   // 128
    static constexpr int Block_N = BlockShape::N;   // 64
    static constexpr int Block_K = BlockShape::K;   // 32
    static constexpr int Warp_M = WarpShape::M;     // 32
    static constexpr int Warp_N = WarpShape::N;     // 64
    static constexpr int Warp_K = WarpShape::K;     // 32
    static constexpr int Mma_M = MmaShape::M;       // 16
    static constexpr int Mma_N = MmaShape::N;       // 8
    static constexpr int Mma_K = MmaShape::K;       // 32
    static constexpr int NStage = ThreadBlockStage;

    static constexpr int indicesPrefetchBlock = 512;

    static constexpr size_t input_buffer_size =
            Block_M * Block_K * SPTC_N / SPTC_M * ThreadBlockStage * sizeof(half) +
            Block_K * Block_N * ThreadBlockStage * sizeof(half) +
            indicesPrefetchBlock * A_INDICES_BUFFER_NUM * sizeof(uint) +
            Block_N * sizeof(uint);
    static constexpr size_t output_buffer_size =
            Block_M / N * M * Block_N * sizeof(half);

    typedef A_fragment<MmaShape> A_Fragment;
    typedef B_fragment<MmaShape> B_Fragment;
    typedef C_fragment<MmaShape, AccumulatorType> C_Fragment;
    typedef Meta_fragment<MmaShape> Meta_Fragment;

    __device__ __forceinline__
    void mainLoop(const int m, const int n, const int k,
                  const int brow, const int mbrow,
                  const half *A_values, const uint *A_metadata, const uint *A_indices,
                  const half *B, const uint *B_indices, const uint B_indices_len,
                  half *shared_mem_workspace);

    __device__ __forceinline__
    void mainLoopTrans(const int m, const int n, const int k,
                  const int brow, const int mbrow,
                  const half *A_values, const uint *A_metadata, const uint *A_indices,
                  const half *B, const uint *B_indices, const uint B_indices_len,
                  half *shared_mem_workspace, const bool silu=false);

    __device__ __forceinline__
    void epilogue(const int m, const int n,
                  // const int N, const int M,
                  const uint B_indices_len,
                  const half *C, half *D, half *shared_mem_workspace);

    __device__ __forceinline__
    void epilogueTrans(const int m, const int n,
                  const uint B_indices_len,
                  half *D, half *shared_mem_workspace);
};

template<
        typename SparseRatio, int VectorLength,
        typename BlockShape, typename WarpShape, typename MmaShape,
        int ThreadBlockStage, typename AccumulatorType,
        typename ASwizzle, typename BSwizzle, typename CSwizzle
>
__device__ __forceinline__
void HorizontalSsmmKernel<SparseRatio, VectorLength, BlockShape, WarpShape, MmaShape,
        ThreadBlockStage, AccumulatorType,
        ASwizzle, BSwizzle, CSwizzle>::mainLoop
        (const int m, const int n, const int k,
         // const int vector_length, const int N, const int M,
         const int brow, const int mbrow,
         const half *A_values, const uint *A_metadata, const uint *A_indices,
         const half *B, const uint *B_indices, const uint B_indices_len,
         half *shared_mem_workspace) {

    const int warp_id = threadIdx.x / THREADS_PER_WARP;
    const int lane_id = threadIdx.x % THREADS_PER_WARP;

    // 计算实际的A相关数据
    const int values_num_cols = k / SPTC_M * SPTC_N;
    const int metadata_num_cols = values_num_cols / NUM_OF_META_PER_UINT;
    // A_indice是转置的
    const int A_indices_num_rows = k / vector_length; // A_indices_row 代表有几组vector_length粒度的剪裁
    const int A_indices_num_cols = m / M * N; // indices_col 代表condense_A的行数

    const int block_idx_M = blockIdx.x;
    const int block_idx_N = blockIdx.y;
    const int warp_idx_M = warp_id % (Block_M / Warp_M);
    const int warp_idx_N = warp_id / (Block_M / Warp_M);

    // 计算当前block需要处理的数据(共64次计算)在GM中的偏移(以thread为单位的偏移)
    const half *A_panel = &A_values[
            (block_idx_M * Block_M + threadIdx.x / A_GM_TO_SM_CP_THREAD_PER_ROW) * values_num_cols +
            threadIdx.x % A_GM_TO_SM_CP_THREAD_PER_ROW * A_GM_TO_SM_CP_SIZE];
    const uint *A_I_panel = &A_indices[block_idx_M * Block_M];
    const uint *M_panel = &A_metadata[(block_idx_M * Block_M) * metadata_num_cols];
    // B数据的具体偏移需要根据B_indices来计算
    // const half *B_panel = &B[(block_idx_N * Block_N) * k];
    const uint *B_I_panel = &B_indices[block_idx_N * Block_N];

    // 计算SM中A、B块的大小占用
    const int A_tile_size = Block_M * Block_K * SPTC_N / SPTC_M;
    const int B_tile_size = Block_K * Block_N;

    // 计算A、B、A_I、B_I在SM中的偏移
    half *A_shared = shared_mem_workspace;
    half *B_shared = A_shared + A_tile_size * ThreadBlockStage;

    uint *A_indices_shared = (uint *) (B_shared + B_tile_size * ThreadBlockStage);
    uint *B_indices_shared = A_indices_shared + indicesPrefetchBlock * A_INDICES_BUFFER_NUM;

    const int mma_iter_M = Warp_M / Mma_M;
    const int mma_iter_N = Warp_N / Mma_N;
    A_Fragment aFragment[mma_iter_M];
    B_Fragment bFragment[mma_iter_N];
    C_Fragment cFragment[mma_iter_M][mma_iter_N][3];
    Meta_Fragment metaFragment[mma_iter_M];

    Pipeline<NStage, true> pipeline;

    ASwizzle aSwizzle;
    BSwizzle bSwizzle;
    CSwizzle cSwizzle;

    uint2 selected_A_Indices[mma_iter_M];

    // 加载B_I到SM
    // 在一个block的计算中，B_I始终只加载一份（不同tile的B_I是相同的），这里的*2是因为每个uint可以存放2个half
    // Block_N>=64, 以64为单位重复加载
    // 当Block_N=64时，每个线程加载Block_N/TOTAL_THREADS_PER_BLOCK个uint(2 byte = 1 half)
    for (int i = 0; i < 2 * Block_N / TOTAL_THREADS_PER_BLOCK; i++) {
        int offset = threadIdx.x * Block_N * sizeof(uint) / sizeof(half) / TOTAL_THREADS_PER_BLOCK + i;
        *(((half *) B_indices_shared) + offset) = *(((half *) B_I_panel) + offset);
    }
    __syncthreads();

    // A_indices的加载，每prefetchIndicesStage个tile加载一次
    const int prefetchIndicesStage = indicesPrefetchBlock / Block_M * vector_length / Block_K;
    int fetch_indices = 0;

    const int num_tiles = k / Block_K;
    int fetch = 0;

#pragma unroll
    for (int compute = 0; compute < num_tiles; compute++) {
        // load metadata to Fragment(GM->Fragment)
        for (int i = 0; i < mma_iter_M; i++) {
            // 这里与论文中的设计有所不同，在16行中，顺序应当是0、4、8、...、28、1、5、9、...、29
            load_meta_gm_to_frag_sync_uint(metaFragment[i].x,
                                            M_panel + i * A_GM_TO_SM_CP_ROWS_PER_ITER * metadata_num_cols +
                                            warp_id * 16 * metadata_num_cols + lane_id / 4 * metadata_num_cols +
                                            lane_id % 2 * 8 * metadata_num_cols, compute);
            // load_meta_gm_to_frag_sync_short(metaFragment[i].x,
            //                                M_panel + i * A_GM_TO_SM_CP_ROWS_PER_ITER * metadata_num_cols +
            //                                warp_id * 16 * metadata_num_cols + lane_id / 4 * metadata_num_cols,
            //                                compute, (lane_id % 4) % 2, metadata_num_cols);
        }

#pragma unroll
        // prefetch A B A_I from GM to SM
        for (; fetch < compute + NStage; fetch++) {
            pipeline.acquire_writer();

            // load A_I from GM to SM
            // prefetch A_indices (used when loading C from SM to Fragment)
            if (fetch % prefetchIndicesStage == 0) {
                if (fetch_indices < (A_indices_num_rows * Block_M / indicesPrefetchBlock)) {
                    const uint *I_tile = A_I_panel + fetch_indices * (indicesPrefetchBlock / Block_M) * A_indices_num_cols+
                                         threadIdx.x * (indicesPrefetchBlock / TOTAL_THREADS_PER_BLOCK) /
                                         Block_M * A_indices_num_cols +
                                         threadIdx.x * (indicesPrefetchBlock / TOTAL_THREADS_PER_BLOCK) %
                                         Block_M;
                    uint *A_I_shared_tile =
                            A_indices_shared + (fetch_indices % A_INDICES_BUFFER_NUM) * indicesPrefetchBlock
                            + threadIdx.x * (indicesPrefetchBlock / TOTAL_THREADS_PER_BLOCK);
                    cp_gm_to_sm_async_zfill<indicesPrefetchBlock / TOTAL_THREADS_PER_BLOCK * sizeof(uint)>(
                            A_I_shared_tile, I_tile);
                }
                fetch_indices++;
            }

            if (fetch < num_tiles) {
                half *shared_tile_A = A_shared + (fetch % NStage) * A_tile_size;
                half *shared_tile_B = B_shared + (fetch % NStage) * B_tile_size;

                // load A from GM to SM
                // 每个线程加载A_tile_size/TOTAL_THREADS_PER_BLOCK个half，一次加载A_GM_TO_SM_CP_SIZE个half
                const int A_copy_iter = A_tile_size / TOTAL_THREADS_PER_BLOCK / A_GM_TO_SM_CP_SIZE;
#pragma unroll
                for (int i = 0; i < A_copy_iter; i++) {
                    // 每次fetch向右偏移A_tile_size / Block_M = Block_K / 2
                    const half *src =
                            A_panel + fetch * A_tile_size / Block_M + i * A_GM_TO_SM_CP_ROWS_PER_ITER * values_num_cols;
                    half *dst = shared_tile_A + aSwizzle(threadIdx.x * A_tile_size / A_copy_iter / TOTAL_THREADS_PER_BLOCK +
                                                         i * A_GM_TO_SM_CP_ROWS_PER_ITER * Block_K * SPTC_N / SPTC_M);
                    // 拷贝时以byte为单位，但是dst和src的指针是half类型
                    // 一次拷贝16个byte，相当于8个half
                    cp_gm_to_sm_async_zfill<A_GM_TO_SM_CP_SIZE * sizeof(half)>(dst, src);
                }

                // load B from GM to SM
                const int B_copy_iter = B_tile_size / TOTAL_THREADS_PER_BLOCK / B_GM_TO_SM_CP_SIZE;
#pragma unroll
                for (int i = 0; i < B_copy_iter; i++) {
                    const int B_row_offset = B_indices_shared[B_GM_TO_SM_CP_ROWS_PER_ITER * i +
                                                              threadIdx.x / B_GM_TO_SM_CP_THREAD_PER_ROW];
                    // A*BT中B的加载（加载转置的B）
                    // 在BT中每次fetch向右偏移Block_K
                    const half *src = &B[fetch * Block_K + B_row_offset * k +
                                         threadIdx.x % B_GM_TO_SM_CP_THREAD_PER_ROW * B_GM_TO_SM_CP_SIZE];
                    int dst_offset = (threadIdx.x / B_GM_TO_SM_CP_THREAD_PER_ROW + B_GM_TO_SM_CP_ROWS_PER_ITER * i) * Block_K +
                                     threadIdx.x % B_GM_TO_SM_CP_THREAD_PER_ROW * B_GM_TO_SM_CP_SIZE;
#pragma unroll
                    for (int j = 0; j < B_GM_TO_SM_CP_TIMES; j++) {
                        cp_gm_to_sm_async_zfill<B_GM_TO_SM_CP_SIZE / B_GM_TO_SM_CP_TIMES * sizeof(half)>(
                                shared_tile_B + bSwizzle(dst_offset + j * B_GM_TO_SM_CP_SIZE / B_GM_TO_SM_CP_TIMES),
                                src + j * B_GM_TO_SM_CP_SIZE / B_GM_TO_SM_CP_TIMES);
                    }
                }
            }
            pipeline.commit_stage();
        }
        pipeline.acquire_reader();

        half *shared_tile_A = A_shared + (compute % NStage) * A_tile_size;
        half *shared_tile_B = B_shared + (compute % NStage) * B_tile_size;
        uint *shared_A_I =
                A_indices_shared + (compute / prefetchIndicesStage % A_INDICES_BUFFER_NUM) * indicesPrefetchBlock;

        // load A from SM to Fragment
#pragma unroll
        for (int i = 0; i < mma_iter_M; i++) {
            load_matrix_sm_to_frag_sync_no_trans<ASwizzle>(aFragment[i], shared_tile_A,
                                                           (i * A_GM_TO_SM_CP_ROWS_PER_ITER + warp_idx_M * Mma_M) *
                                                           Block_K * SPTC_N / SPTC_M,
                                                           Block_K * SPTC_N / SPTC_M, lane_id);
        }

        // load B from SM to Fragment
#pragma unroll
        for (int i = 0; i < mma_iter_N; i++) {
            // 数据加载的位置需要进一步确认
            load_matrix_sm_to_frag_sync_no_trans<BSwizzle>(bFragment[i], shared_tile_B,
                                                           warp_idx_N * Warp_N + i * Mma_N * Mma_K, Block_K, lane_id);
        }

        if (compute % (vector_length / Block_K) == 0) {
#pragma unroll
            for (int i = 0; i < mma_iter_M; ++i) {
                int temp =
                        (compute % prefetchIndicesStage) / (prefetchIndicesStage / (indicesPrefetchBlock / Block_M)) *
                        Block_M + A_INDICES_GM_TO_SM_CP_COLS_PER_ITER * i + warp_id * Mma_M + lane_id / 4;
                selected_A_Indices[i].x = shared_A_I[temp];
                selected_A_Indices[i].y = shared_A_I[temp + 8];
            }
#pragma unroll
            for (int i = 0; i < mma_iter_M; i++) {
                for (int j = 0; j < mma_iter_N; j++) {
                    if (!selected_A_Indices[i].x) {
                        mv_cfrag_x_p1(cFragment[i][j][1], cFragment[i][j][0]);
                    } else {
                        mv_cfrag_x_p1(cFragment[i][j][2], cFragment[i][j][0]);
                    }
                    if (!selected_A_Indices[i].y) {
                        mv_cfrag_x_p2(cFragment[i][j][1], cFragment[i][j][0]);
                    } else {
                        mv_cfrag_x_p2(cFragment[i][j][2], cFragment[i][j][0]);
                    }
                }
            }
        }

#pragma unroll
        for (int i = 0; i < mma_iter_M; i++) {
#pragma unroll
            for (int j = 0; j < mma_iter_N; j++) {
                mma_sync_sparse(cFragment[i][j][0], aFragment[i], bFragment[j], cFragment[i][j][0], metaFragment[i]);

            }
        }
        
        if ((compute + 1) % (vector_length / Block_K) == 0) {
#pragma unroll
            for (int i = 0; i < mma_iter_M; i++) {
                for (int j = 0; j < mma_iter_N; j++) {
                    if (!selected_A_Indices[i].x) {
                        mv_cfrag_x_p1(cFragment[i][j][0], cFragment[i][j][1]);
                    } else {
                        mv_cfrag_x_p1(cFragment[i][j][0], cFragment[i][j][2]);
                    }
                    if (!selected_A_Indices[i].y) {
                        mv_cfrag_x_p2(cFragment[i][j][0], cFragment[i][j][1]);
                    } else {
                        mv_cfrag_x_p2(cFragment[i][j][0], cFragment[i][j][2]);
                    }
                }
            }
        }
        pipeline.release_reader();
    }

    // 把C加载到SM中
    half *shared_C = shared_mem_workspace;
    // 包含warp种每个thread的偏移
    int C_warp_tile_offset = warp_idx_M * Mma_M / N * M * Block_N + lane_id / 4 / N * M * Block_N + (lane_id % 4) * 2;

    __syncthreads();

#pragma unroll
    for (int i = 0; i < mma_iter_M; i++) {
        int offset = C_warp_tile_offset + i * C_SM_TO_GM_CP_ROWS_PER_ITER * Block_N;
#pragma unroll
        for (int j = 0; j < mma_iter_N; j++) {
            store_matrix_frag_to_sm<CSwizzle>(cFragment[i][j][1], shared_C, offset + j * Mma_N, Block_N);
            store_matrix_frag_to_sm<CSwizzle>(cFragment[i][j][2], shared_C, offset + Block_N + j * Mma_N, Block_N);
        }
    }
}

template<
        typename SparseRatio, int VectorLength,
        typename BlockShape, typename WarpShape, typename MmaShape,
        int ThreadBlockStage, typename AccumulatorType,
        typename ASwizzle, typename BSwizzle, typename CSwizzle
>
__device__ __forceinline__
void HorizontalSsmmKernel<SparseRatio, VectorLength, BlockShape, WarpShape, MmaShape,
        ThreadBlockStage, AccumulatorType,
        ASwizzle, BSwizzle, CSwizzle>::mainLoopTrans
        (const int m, const int n, const int k,
         const int brow, const int mbrow,
         const half *A_values, const uint *A_metadata, const uint *A_indices,
         const half *B, const uint *B_indices, const uint B_indices_len,
         half *shared_mem_workspace, const bool silu) {

    const int warp_id = threadIdx.x / THREADS_PER_WARP;
    const int lane_id = threadIdx.x % THREADS_PER_WARP;

    // 计算实际的A相关数据
    const int values_num_cols = k / SPTC_M * SPTC_N;
    const int metadata_num_cols = values_num_cols / NUM_OF_META_PER_UINT;
    // A_indice是转置的
    const int A_indices_num_rows = k / vector_length; // A_indices_row 代表有几组vector_length粒度的剪裁
    const int A_indices_num_cols = m / M * N; // indices_col 代表condense_A的行数

    const int block_idx_M = blockIdx.x;
    const int block_idx_N = blockIdx.y;
    const int warp_idx_M = warp_id % (Block_M / Warp_M);
    const int warp_idx_N = warp_id / (Block_M / Warp_M);

    // 计算当前block需要处理的数据(共64次计算)在GM中的偏移(以thread为单位的偏移)
    const half *A_panel = &A_values[
            (block_idx_M * Block_M + threadIdx.x / A_GM_TO_SM_CP_THREAD_PER_ROW) * values_num_cols +
            threadIdx.x % A_GM_TO_SM_CP_THREAD_PER_ROW * A_GM_TO_SM_CP_SIZE];
    const uint *A_I_panel = &A_indices[block_idx_M * Block_M];
    const uint *M_panel = &A_metadata[(block_idx_M * Block_M) * metadata_num_cols];
    // B数据的具体偏移需要根据B_indices来计算
    // const half *B_panel = &B[(block_idx_N * Block_N) * k];
    const uint *B_I_panel = &B_indices[block_idx_N * Block_N];

    // 计算SM中A、B块的大小占用
    const int A_tile_size = Block_M * Block_K * SPTC_N / SPTC_M;
    const int B_tile_size = Block_K * Block_N;

    // 计算A、B、A_I、B_I在SM中的偏移
    half *A_shared = shared_mem_workspace;
    half *B_shared = A_shared + A_tile_size * ThreadBlockStage;

    uint *A_indices_shared = (uint *) (B_shared + B_tile_size * ThreadBlockStage);
    uint *B_indices_shared = A_indices_shared + indicesPrefetchBlock * A_INDICES_BUFFER_NUM;

    const int mma_iter_M = Warp_M / Mma_M;
    const int mma_iter_N = Warp_N / Mma_N;
    A_Fragment aFragment[mma_iter_M];
    B_Fragment bFragment[mma_iter_N];
    C_Fragment cFragment[mma_iter_M][mma_iter_N][3];
    Meta_Fragment metaFragment[mma_iter_M];

    Pipeline<NStage, true> pipeline;

    ASwizzle aSwizzle;
    BSwizzle bSwizzle;
    CSwizzle cSwizzle;

    uint2 selected_A_Indices[mma_iter_M];

    // 加载B_I到SM
    // 在一个block的计算中，B_I始终只加载一份（不同tile的B_I是相同的），这里的*2是因为每个uint可以存放2个half
    // Block_N>=64, 以64为单位重复加载
    // 当Block_N=64时，每个线程加载Block_N/TOTAL_THREADS_PER_BLOCK个uint(2 byte = 1 half)
    for (int i = 0; i < 2 * Block_N / TOTAL_THREADS_PER_BLOCK; i++) {
        int offset = threadIdx.x * Block_N * sizeof(uint) / sizeof(half) / TOTAL_THREADS_PER_BLOCK + i;
        *(((half *) B_indices_shared) + offset) = *(((half *) B_I_panel) + offset);
    }
    __syncthreads();

    // A_indices的加载，每prefetchIndicesStage个tile加载一次
    const int prefetchIndicesStage = indicesPrefetchBlock / Block_M * vector_length / Block_K;
    int fetch_indices = 0;

    const int num_tiles = k / Block_K;
    int fetch = 0;

#pragma unroll
    for (int compute = 0; compute < num_tiles; compute++) {
        // load metadata to Fragment(GM->Fragment)
        for (int i = 0; i < mma_iter_M; i++) {
            load_meta_gm_to_frag_sync_uint(metaFragment[i].x,
                                           M_panel + i * A_GM_TO_SM_CP_ROWS_PER_ITER * metadata_num_cols +
                                           warp_id * 16 * metadata_num_cols + lane_id / 4 * metadata_num_cols,
                                           compute);
        }

#pragma unroll
        // prefetch A B A_I from GM to SM
        for (; fetch < compute + NStage; fetch++) {
            pipeline.acquire_writer();

            // load A_I from GM to SM
            // prefetch A_indices (used when loading C from SM to Fragment)
            if (fetch % prefetchIndicesStage == 0) {
                if (fetch_indices < (A_indices_num_rows * Block_M / indicesPrefetchBlock)) {
                    const uint *I_tile = A_I_panel + fetch_indices * (indicesPrefetchBlock / Block_M) * A_indices_num_cols+
                                         threadIdx.x * (indicesPrefetchBlock / TOTAL_THREADS_PER_BLOCK) /
                                         Block_M * A_indices_num_cols +
                                         threadIdx.x * (indicesPrefetchBlock / TOTAL_THREADS_PER_BLOCK) %
                                         Block_M;
                    uint *A_I_shared_tile =
                            A_indices_shared + (fetch_indices % A_INDICES_BUFFER_NUM) * indicesPrefetchBlock
                            + threadIdx.x * (indicesPrefetchBlock / TOTAL_THREADS_PER_BLOCK);
                    cp_gm_to_sm_async_zfill<indicesPrefetchBlock / TOTAL_THREADS_PER_BLOCK * sizeof(uint)>(
                            A_I_shared_tile, I_tile);
                }
                fetch_indices++;
            }

            if (fetch < num_tiles) {
                half *shared_tile_A = A_shared + (fetch % NStage) * A_tile_size;
                half *shared_tile_B = B_shared + (fetch % NStage) * B_tile_size;

                // load A from GM to SM
                // 每个线程加载A_tile_size/TOTAL_THREADS_PER_BLOCK个half，一次加载A_GM_TO_SM_CP_SIZE个half
                const int A_copy_iter = A_tile_size / TOTAL_THREADS_PER_BLOCK / A_GM_TO_SM_CP_SIZE;
#pragma unroll
                for (int i = 0; i < A_copy_iter; i++) {
                    // 每次fetch向右偏移A_tile_size / Block_M = Block_K / 2
                    const half *src =
                            A_panel + fetch * A_tile_size / Block_M + i * A_GM_TO_SM_CP_ROWS_PER_ITER * values_num_cols;
                    half *dst = shared_tile_A + aSwizzle(threadIdx.x * A_tile_size / A_copy_iter / TOTAL_THREADS_PER_BLOCK +
                                                         i * A_GM_TO_SM_CP_ROWS_PER_ITER * Block_K * SPTC_N / SPTC_M);
                    // 拷贝时以byte为单位，但是dst和src的指针是half类型
                    // 一次拷贝16个byte，相当于8个half
                    cp_gm_to_sm_async_zfill<A_GM_TO_SM_CP_SIZE * sizeof(half)>(dst, src);
                }

                // load B from GM to SM
                const int B_copy_iter = B_tile_size / TOTAL_THREADS_PER_BLOCK / B_GM_TO_SM_CP_SIZE;
#pragma unroll
                for (int i = 0; i < B_copy_iter; i++) {
                    const int B_row_offset = B_indices_shared[B_GM_TO_SM_CP_ROWS_PER_ITER * i +
                                                              threadIdx.x / B_GM_TO_SM_CP_THREAD_PER_ROW];
                    // A*BT中B的加载（加载转置的B）
                    // 在BT中每次fetch向右偏移Block_K
                    const half *src = &B[fetch * Block_K + B_row_offset * k +
                                         threadIdx.x % B_GM_TO_SM_CP_THREAD_PER_ROW * B_GM_TO_SM_CP_SIZE];
                    int dst_offset = (threadIdx.x / B_GM_TO_SM_CP_THREAD_PER_ROW + B_GM_TO_SM_CP_ROWS_PER_ITER * i) * Block_K +
                                     threadIdx.x % B_GM_TO_SM_CP_THREAD_PER_ROW * B_GM_TO_SM_CP_SIZE;
#pragma unroll
                    for (int j = 0; j < B_GM_TO_SM_CP_TIMES; j++) {
                        cp_gm_to_sm_async_zfill<B_GM_TO_SM_CP_SIZE / B_GM_TO_SM_CP_TIMES * sizeof(half)>(
                                shared_tile_B + bSwizzle(dst_offset + j * B_GM_TO_SM_CP_SIZE / B_GM_TO_SM_CP_TIMES),
                                src + j * B_GM_TO_SM_CP_SIZE / B_GM_TO_SM_CP_TIMES);
                    }
                }
            }
            pipeline.commit_stage();
        }
        pipeline.acquire_reader();

        half *shared_tile_A = A_shared + (compute % NStage) * A_tile_size;
        half *shared_tile_B = B_shared + (compute % NStage) * B_tile_size;
        uint *shared_A_I =
                A_indices_shared + (compute / prefetchIndicesStage % A_INDICES_BUFFER_NUM) * indicesPrefetchBlock;

        // load A from SM to Fragment
#pragma unroll
        for (int i = 0; i < mma_iter_M; i++) {
            load_matrix_sm_to_frag_sync_no_trans<ASwizzle>(aFragment[i], shared_tile_A,
                                                           (i * A_GM_TO_SM_CP_ROWS_PER_ITER + warp_idx_M * Mma_M) *
                                                           Block_K * SPTC_N / SPTC_M,
                                                           Block_K * SPTC_N / SPTC_M, lane_id);
        }

        // load B from SM to Fragment
#pragma unroll
        for (int i = 0; i < mma_iter_N; i++) {
            load_matrix_sm_to_frag_sync_no_trans<BSwizzle>(bFragment[i], shared_tile_B,
                                                           warp_idx_N * Warp_N + i * Mma_N * Mma_K, Block_K, lane_id);
        }

        if (compute % (vector_length / Block_K) == 0) {
#pragma unroll
            for (int i = 0; i < mma_iter_M; ++i) {
                int temp =
                        (compute % prefetchIndicesStage) / (prefetchIndicesStage / (indicesPrefetchBlock / Block_M)) *
                        Block_M + A_INDICES_GM_TO_SM_CP_COLS_PER_ITER * i + warp_id * Mma_M + lane_id / 4;
                selected_A_Indices[i].x = shared_A_I[temp];
                selected_A_Indices[i].y = shared_A_I[temp + 8];
            }
#pragma unroll
            for (int i = 0; i < mma_iter_M; i++) {
                for (int j = 0; j < mma_iter_N; j++) {
                    if (!selected_A_Indices[i].x) {
                        mv_cfrag_x_p1(cFragment[i][j][1], cFragment[i][j][0]);
                    } else {
                        mv_cfrag_x_p1(cFragment[i][j][2], cFragment[i][j][0]);
                    }
                    if (!selected_A_Indices[i].y) {
                        mv_cfrag_x_p2(cFragment[i][j][1], cFragment[i][j][0]);
                    } else {
                        mv_cfrag_x_p2(cFragment[i][j][2], cFragment[i][j][0]);
                    }
                }
            }
        }

#pragma unroll
        for (int i = 0; i < mma_iter_M; i++) {
#pragma unroll
            for (int j = 0; j < mma_iter_N; j++) {
                mma_sync_sparse(cFragment[i][j][0], aFragment[i], bFragment[j], cFragment[i][j][0], metaFragment[i]);
            }
        }

        if ((compute + 1) % (vector_length / Block_K) == 0) {
#pragma unroll
            for (int i = 0; i < mma_iter_M; i++) {
                for (int j = 0; j < mma_iter_N; j++) {
                    if (!selected_A_Indices[i].x) {
                        mv_cfrag_x_p1(cFragment[i][j][0], cFragment[i][j][1]);
                    } else {
                        mv_cfrag_x_p1(cFragment[i][j][0], cFragment[i][j][2]);
                    }
                    if (!selected_A_Indices[i].y) {
                        mv_cfrag_x_p2(cFragment[i][j][0], cFragment[i][j][1]);
                    } else {
                        mv_cfrag_x_p2(cFragment[i][j][0], cFragment[i][j][2]);
                    }
                }
            }
        }
        pipeline.release_reader();
    }

    // 把C加载到SM中
    half *shared_C = shared_mem_workspace;
    // 包含warp种每个thread的偏移
    // 1-3 no trans 加载的起始位置
    // int C_warp_tile_offset = warp_idx_M * Mma_M / N * M * Block_N + lane_id / 4 / N * M * Block_N + (lane_id % 4) * 2;
    // 1-3 trans 加载的起始位置
    int C_warp_tile_offset = warp_idx_M * Mma_M / N * M + lane_id / 4 * 2 + (lane_id % 4) * 2 * Block_M / N * M;
    __syncthreads();

#pragma unroll
    for (int i = 0; i < mma_iter_M; i++) {
        // 2-3 no trans 加载的offset
        // int offset = C_warp_tile_offset + i * C_SM_TO_GM_CP_ROWS_PER_ITER * Block_N;
        // 2-3 trans 加载的offset
        int offset = C_warp_tile_offset + i * C_FRAGMENT_TO_SM_CP_COLS_PER_ITER;
#pragma unroll
        for (int j = 0; j < mma_iter_N; j++) {
            // sm-90支持语法
            // store_matrix_frag_to_sm_sync_no_trans<CSwizzle>(cFragment[i][j][1], cFragment[i][j][2], shared_C, offset + j * Mma_N);
            // 3-3 no trans store
            // store_matrix_frag_to_sm<CSwizzle>(cFragment[i][j][1], shared_C, offset + j * Mma_N, Block_N);
            // store_matrix_frag_to_sm<CSwizzle>(cFragment[i][j][2], shared_C, offset + Block_N + j * Mma_N, Block_N);

            // 3-3 trans store
            half tmp = ((half *) &(cFragment[i][j][1].x_p1[0]))[1];
            ((half *) &(cFragment[i][j][1].x_p1[0]))[1] = ((half *) &(cFragment[i][j][2].x_p1[0]))[0];
            ((half *) &(cFragment[i][j][2].x_p1[0]))[0] = tmp;

            tmp = ((half *) &(cFragment[i][j][1].x_p2[0]))[1];
            ((half *) &(cFragment[i][j][1].x_p2[0]))[1] = ((half *) &(cFragment[i][j][2].x_p2[0]))[0];
            ((half *) &(cFragment[i][j][2].x_p2[0]))[0] = tmp;

            store_matrix_frag_to_sm<CSwizzle>(cFragment[i][j][1], shared_C, offset + j * Mma_N * Block_M / N * M, silu);
            store_matrix_frag_to_sm<CSwizzle>(cFragment[i][j][2], shared_C, offset + (1 + j * Mma_N) * Block_M / N * M, silu);
        }
    }
}

template<
        typename SparseRatio, int VectorLength,
        typename BlockShape, typename WarpShape, typename MmaShape,
        int ThreadBlockStage, typename AccumulatorType,
        typename ASwizzle, typename BSwizzle, typename CSwizzle
>
__device__ __forceinline__
void HorizontalSsmmKernel<SparseRatio, VectorLength, BlockShape, WarpShape, MmaShape,
        ThreadBlockStage, AccumulatorType,
        ASwizzle, BSwizzle, CSwizzle>::epilogue
        (const int m, const int n,
         // const int N, const int M,
         const uint B_indices_len,
         const half *C, half *D, half *shared_mem_workspace) {
    // load C from SM to GM
    const int block_idx_M = blockIdx.x;
    const int block_idx_N = blockIdx.y;

    CSwizzle cSwizzle;

    half *D_tile = D + block_idx_M * Block_M / N * M * B_indices_len + block_idx_N * Block_N;

    half *shared_C = shared_mem_workspace;

#pragma unroll
    for (int i = 0; i < Block_M / N * M / C_SM_TO_GM_CP_ROWS_PER_ITER; ++i) {
#pragma unroll
        for (int j = 0; j < Block_N / C_SM_TO_GM_CP_COLS_PER_ITER; ++j) {
#pragma unroll
            for (int iter = 0; iter < C_SM_TO_GM_CP_TIMES; ++iter) {
                int row_offset = i * C_SM_TO_GM_CP_ROWS_PER_ITER + threadIdx.x;
                int col_offset = j * C_SM_TO_GM_CP_SIZE + iter * C_SM_TO_GM_CP_SIZE / C_SM_TO_GM_CP_TIMES;
                half *src = shared_C + cSwizzle(row_offset * Block_N + col_offset);
                half *dst = D_tile + row_offset * B_indices_len + col_offset;
                *((float4 *) dst) = *((float4 *) src);
            }
        }
    }
}

template<
        typename SparseRatio, int VectorLength,
        typename BlockShape, typename WarpShape, typename MmaShape,
        int ThreadBlockStage, typename AccumulatorType,
        typename ASwizzle, typename BSwizzle, typename CSwizzle
>
__device__ __forceinline__
void HorizontalSsmmKernel<SparseRatio, VectorLength, BlockShape, WarpShape, MmaShape,
        ThreadBlockStage, AccumulatorType,
        ASwizzle, BSwizzle, CSwizzle>::epilogueTrans
        (const int m, const int n,
         const uint B_indices_len,
         half *D, half *shared_mem_workspace){
    // 返回转置的result，大小为B_indices_len*m
    // load C from SM to GM
    const int block_idx_M = blockIdx.x;
    const int block_idx_N = blockIdx.y;

    CSwizzle cSwizzle;

    half *D_tile = D + block_idx_M * Block_M / N * M;

    half *shared_C = shared_mem_workspace;

#pragma unroll
    for (int i = 0; i < Block_N / C_SM_TO_GM_CP_COLS_PER_ITER; ++i) {
#pragma unroll
        for (int j = 0; j < Block_M / N * M / C_SM_TO_GM_CP_ROWS_PER_ITER; ++j) {
            int C_row_offset = threadIdx.x % Block_N;
            int D_row_offset = C_row_offset + block_idx_N * Block_N;
#pragma unroll
            for (int iter = 0; iter < C_SM_TO_GM_CP_TIMES; ++iter) {
                int col_offset = threadIdx.x / Block_N * C_SM_TO_GM_CP_SIZE + iter * C_SM_TO_GM_CP_SIZE / C_SM_TO_GM_CP_TIMES;
                half *src = shared_C + cSwizzle(C_row_offset * Block_M / N * M + col_offset);
                half *dst = D_tile + D_row_offset * m + col_offset;
                *((float4 *) dst) = *((float4 *) src);
            }
        }
    }
}
#endif //FLEXIBLE_SPMM_HORIZONTAL_SSMM_KERNEL_H
