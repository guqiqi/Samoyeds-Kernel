//
// Created by CPWu2017@126.com on 12/22/23.
//
#include <iostream>
#include <chrono>
#include <random>
#include "./util/argparse.hpp"

#include "./format/formats.hpp"
#include "./ssmm/horizontal_ssmm_kernel_op.h"
#include "./util/matrix_utils.h"
#include "./util/print_helper.hpp"
#include "./util/result_check.h"

// #include "./gemm/cublasLt_gemm.h"
#include "./gemm/cublas_gemm.h"

template <typename T>
void launch_kernels(int m, int k, int n, int V, int N, int M, int vector_length, uint seed, bool check, bool measure_time, bool save_result)
{
    // auto *format = new Format_venom<T>(m, k, seed, V, N, M, vector_length);
    // auto *format = new Format_ss<T>(m, k seed, vector_length, N, M);
    auto *format = new Format_ss<T>(m, k, seed, vector_length, N, M);
    format->init();
    format->sync_device();

    // int dense_B_lens = n / 4;
    // VerticalSparseMatrix<T> B;
    int dense_B_lens = n / 4;
    HorizontalSparseMatrix<T> B;
    ZerosMatrix<T> C;
    // int dense_B_cols = n / 4;
    // HorizontalSparseMatrix是转置的矩阵, 所以行列数是相反的
    B.init_DataMatrix(n, k, dense_B_lens, true);
    // 4-1 HorizontalSsmmKernelExec初始化C
    // C.init_DataMatrix(m, dense_B_lens);
    // 4-1 HorizontalSsmmTransKernelExec初始化C
    C.init_DataMatrix(dense_B_lens, m);

    using BlockShape = ShapeBase<128, 32, 64>;
    using WarpShape = ShapeBase<32, 32, 64>;
    using MmaShape = ShapeBase<16, 32, 8>;

    // 4-1 HorizontalSsmmKernelExec执行
    HorizontalSsmmKernelExec<BlockShape, WarpShape, MmaShape, 2>(
            m, n, k, vector_length, N, M,
            format->devicePtr._value,
            format->devicePtr._meta,
            format->devicePtr._indices,
            B.device_ptr, B.indices_device_ptr, B.indices_len, C.device_ptr, C.device_ptr);
    // 4-2 HorizontalSsmmTransKernelExec执行
    // HorizontalSsmmTransKernelExec<BlockShape, WarpShape, MmaShape, 2>(
    //         m, n, k, vector_length, N, M,
    //         format->devicePtr._value,
    //         format->devicePtr._meta,
    //         format->devicePtr._indices,
    //         B.device_ptr, B.indices_device_ptr, B.indices_len, C.device_ptr, C.device_ptr);
    // 4-2 HorizontalSsmmFusedActTransKernelExec执行
    // HorizontalSsmmFusedActTransKernelExec<BlockShape, WarpShape, MmaShape, 2>(
    //         m, n, k, vector_length, N, M,
    //         format->devicePtr._value,
    //         format->devicePtr._meta,
    //         format->devicePtr._indices,
    //         B.device_ptr, B.indices_device_ptr, B.indices_len, C.device_ptr, C.device_ptr);

    C.sync_host();
    if (save_result)
    {
        // write_to_file_dec(format->data, format->rows, format->cols, "A_data");
        // write_to_file_dec(format->indices, format->get_num_of_rows_indices(), format->get_num_of_cols_indices(),
        //                   "A_indices");
        // write_to_file_hex(format->metadata, format->get_num_of_rows_metadata(), format->get_num_of_cols_metadata(),
        //                   "A_metadata");
        // write_to_file_dec(format->value, format->get_num_of_rows_value(), format->get_num_of_cols_value(),
        //                   "A_value", 256, 64);
        // write_to_file_dec(B.host_ptr, B.rows, B.cols, "B", 128, 64);
        write_to_file_dec(C.host_ptr, C.rows, C.cols, "C", 1024, 16);
    }

    if (check)
    {   // GEMM计算
        // A: format->data B: B.device_ptr C_gemm: 结果
        ZerosMatrix<T> C_gemm;
        C_gemm.init_DataMatrix(m, dense_B_lens);

        // 计算
        GemmExec(m, dense_B_lens, k, format->devicePtr._pruned_value, B.condense_device_ptr, C_gemm.device_ptr);

        C_gemm.sync_host();

        if (save_result) {
            // write_to_file_dec(B.condense_host_ptr, B.condense_rows, B.condense_cols, "B_dense", 128, 64);
            // write_to_file_dec(format->pruned_value, format->get_num_of_rows_pruned_value(), format->get_num_of_cols_pruned_value(), "A_gemm", 128, 128);
            write_to_file_dec(C_gemm.host_ptr, C_gemm.rows, C_gemm.cols, "C_gemm", 1024, 16);
        }

        // 对比结果 C.host_ptr, C_gemm.host_ptr
        auto error = check_results(C.host_ptr, C_gemm.host_ptr, C.rows, C.cols);
        std::cout << "compare result: " << std::boolalpha << error << std::endl;
    }

    if (measure_time) {
        for (int i = 0; i < 10; i++) {
            // 4-3 HorizontalSsmmKernelExec执行
            HorizontalSsmmKernelExec<BlockShape, WarpShape, MmaShape, 2>(
                    m, n, k, vector_length, N, M,
                    format->devicePtr._value,
                    format->devicePtr._meta,
                    format->devicePtr._indices,
                    B.device_ptr, B.indices_device_ptr, B.indices_len, C.device_ptr, C.device_ptr);
            // 4-3 HorizontalSsmmTransKernelExec执行
            // HorizontalSsmmTransKernelExec<BlockShape, WarpShape, MmaShape, 2>(
            //         m, n, k, vector_length, N, M,
            //         format->devicePtr._value,
            //         format->devicePtr._meta,
            //         format->devicePtr._indices,
            //         B.device_ptr, B.indices_device_ptr, B.indices_len, C.device_ptr, C.device_ptr);
            // 4-3 HorizontalSsmmFusedActTransKernelExec执行
            // HorizontalSsmmFusedActTransKernelExec<BlockShape, WarpShape, MmaShape, 2>(
            //         m, n, k, vector_length, N, M,
            //         format->devicePtr._value,
            //         format->devicePtr._meta,
            //         format->devicePtr._indices,
            //         B.device_ptr, B.indices_device_ptr, B.indices_len, C.device_ptr, C.device_ptr);
        }
        // Get the start time
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 10000; i++) {
            // 4-4 HorizontalSsmmKernelExec执行
            HorizontalSsmmKernelExec<BlockShape, WarpShape, MmaShape, 2>(
                    m, n, k, vector_length, N, M,
                    format->devicePtr._value,
                    format->devicePtr._meta,
                    format->devicePtr._indices,
                    B.device_ptr, B.indices_device_ptr, B.indices_len, C.device_ptr, C.device_ptr);
            // 4-4 HorizontalSsmmTransKernelExec执行
            // HorizontalSsmmTransKernelExec<BlockShape, WarpShape, MmaShape, 2>(
            //         m, n, k, vector_length, N, M,
            //         format->devicePtr._value,
            //         format->devicePtr._meta,
            //         format->devicePtr._indices,
            //         B.device_ptr, B.indices_device_ptr, B.indices_len, C.device_ptr, C.device_ptr);
            // 4-4 HorizontalSsmmFusedActTransKernelExec执行
            // HorizontalSsmmFusedActTransKernelExec<BlockShape, WarpShape, MmaShape, 2>(
            //         m, n, k, vector_length, N, M,
            //         format->devicePtr._value,
            //         format->devicePtr._meta,
            //         format->devicePtr._indices,
            //         B.device_ptr, B.indices_device_ptr, B.indices_len, C.device_ptr, C.device_ptr);

        }
        cudaDeviceSynchronize();
        // Get the end time
        auto end = std::chrono::high_resolution_clock::now();
        // Calculate the duration
        std::chrono::duration<double> diff = end - start;
        // Print the duration
        std::cout << "Time to execute horizontal SSMM kernels 10000 times: " << diff.count() << " s\n";
    }
}

int main(int argc, const char **argv)
{
    argparse::ArgumentParser parser("spmm_bench");
    parser.add_argument("-m").help("Rows number of A").default_value(1024).scan<'i', int>();
    parser.add_argument("-k").help("Cols number of A").default_value(1024).scan<'i', int>();
    parser.add_argument("-n").help("Cols number of B").default_value(16384).scan<'i', int>();
    parser.add_argument("-V").help("V of structural sparsity V:N:M").default_value(8).scan<'i', int>();
    parser.add_argument("-N").help("N of structural sparsity V:N:M").default_value(1).scan<'i', int>();
    parser.add_argument("-M").help("M of structural sparsity V:N:M").default_value(2).scan<'i', int>();
    parser.add_argument("-d", "--density").help("Density of A").default_value(0.5).scan<'g', double>();
    parser.add_argument("--vector_length").help("vector-wise方向上的长度").default_value(32).scan<'i', int>();
    parser.add_argument("-f", "--format").help("Format for A. Can be [venom, blockwise]").default_value("venom");
    parser.add_argument("-s", "--seed").help("Set a random seed, or it is created randomly").scan<'i', int>();
    parser.add_argument("-c", "--check").help("Check the result").flag();
    parser.add_argument("-t", "--time").help("Measure the time consumption").flag();
    parser.add_argument("--save").help("Save the result to csv file").flag();

    parser.parse_args(argc, argv);

    int m = parser.get<int>("-m");
    int k = parser.get<int>("-k");
    int n = parser.get<int>("-n");
    int V = parser.get<int>("-V");
    int N = parser.get<int>("-N");
    int M = parser.get<int>("-M");
    auto density = parser.get<double>("--density");
    int vector_length = parser.get<int>("--vector_length");
    uint seed;
    if (parser.is_used("--seed"))
    {
        seed = parser.get<int>("--seed");
    }
    else
    {
        seed = std::chrono::system_clock::now().time_since_epoch().count();
    }
    std::srand(seed);
    bool check = parser.get<bool>("--check");
    bool measure_time = parser.get<bool>("--time");
    bool save_result = parser.get<bool>("--save");

    launch_kernels<half>(m, k, n, V, N, M, vector_length, seed, check, measure_time, save_result);
}