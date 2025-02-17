//
// Created by CPWu2017@126.com on 12/22/23.
//
#include <iostream>
#include <chrono>
#include <random>
#include "./util/argparse.hpp"

#include "./format/formats.hpp"
#include "./ssmm/horizontal_spmm_kernel_op.h"
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
    // int dense_B_lens = n / 4;
    // SPMM测试
    int dense_B_lens = n;
    HorizontalSparseMatrix<T> B;
    RandomMatrix<T> B_dense;
    ZerosMatrix<T> C;
    ZerosMatrix<T> D;
    ZerosMatrix<T> D_trans;
    // int dense_B_cols = n / 4;
    // HorizontalSparseMatrix是转置的矩阵, 所以行列数是相反的
    B.init_DataMatrix(n, k, dense_B_lens, true);
    B_dense.init_DataMatrix(dense_B_lens, k, true);
    C.init_DataMatrix(m, dense_B_lens);
    D.init_DataMatrix(m, dense_B_lens);
    D_trans.init_DataMatrix(n, m);

    using BlockShape = ShapeBase<128, 32, 64>;
    using WarpShape = ShapeBase<32, 32, 64>;
    using MmaShape = ShapeBase<16, 32, 8>;

    RandomMatrix<T> routing_weights;
    routing_weights.init_DataMatrix(1, dense_B_lens);

    HorizontalSpmmKernelExec<BlockShape, WarpShape, MmaShape, 2>(
            m, dense_B_lens, k, vector_length, N, M,
            format->devicePtr._value,
            format->devicePtr._meta,
            format->devicePtr._indices,
            B_dense.device_ptr,
            C.device_ptr, D_trans.device_ptr);
    D_trans.sync_host();

    // HorizontalSpmmSparseTransKernelExec<BlockShape, WarpShape, MmaShape, 2>(
    //         m, dense_B_lens, k, vector_length, N, M,
    //         format->devicePtr._value,
    //         format->devicePtr._meta,
    //         format->devicePtr._indices,
    //         B_dense.device_ptr, B.indices_device_ptr, B.indices_len, routing_weights.device_ptr, C.device_ptr, D_trans.device_ptr);
    // D_trans.sync_host();

    if (save_result)
    {
        // write_to_file_dec(D.host_ptr, D.rows, D.cols, "D", 16, 128);
        write_to_file_dec(D_trans.host_ptr, D_trans.rows, D_trans.cols, "D_trans", 128, 16);
    }

    if (check)
    {   // GEMM计算
        // A: format->data B: B.device_ptr C_gemm: 结果
        ZerosMatrix<T> C_gemm;
        C_gemm.init_DataMatrix(m, dense_B_lens);

        // 计算
        GemmExec(m, dense_B_lens, k, format->devicePtr._pruned_value, B_dense.transpose_device_ptr, C_gemm.device_ptr);

        C_gemm.sync_host();

        if (save_result) {
            write_to_file_dec(B_dense.host_ptr, B_dense.rows, B_dense.cols, "B_gemm", 64, 128);
            write_to_file_dec(B_dense.transpose_host_ptr, B_dense.cols, B_dense.rows, "B_trans", 128, 64);
            write_to_file_dec(C_gemm.host_ptr, C_gemm.rows, C_gemm.cols, "C_gemm", 16, 128);
        }

        // 对比结果 C.host_ptr, C_gemm.host_ptr
        auto error = check_results(D_trans.host_ptr, C_gemm.host_ptr, C.rows, C.cols);
        std::cout << "compare result: " << std::boolalpha << error << std::endl;
    }

    if (measure_time) {
        for (int i = 0; i < 10; i++) {
            HorizontalSpmmKernelExec<BlockShape, WarpShape, MmaShape, 2>(
                    m, dense_B_lens, k, vector_length, N, M,
                    format->devicePtr._value,
                    format->devicePtr._meta,
                    format->devicePtr._indices,
                    B_dense.device_ptr,
                    C.device_ptr, D_trans.device_ptr);

            // HorizontalSpmmSparseTransKernelExec<BlockShape, WarpShape, MmaShape, 2>(
            //         m, dense_B_lens, k, vector_length, N, M,
            //         format->devicePtr._value,
            //         format->devicePtr._meta,
            //         format->devicePtr._indices,
            //         B_dense.device_ptr, B.indices_device_ptr, B.indices_len, routing_weights.device_ptr, C.device_ptr, D_trans.device_ptr);
        }
        // Get the start time
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 10000; i++) {
            HorizontalSpmmKernelExec<BlockShape, WarpShape, MmaShape, 2>(
                    m, dense_B_lens, k, vector_length, N, M,
                    format->devicePtr._value,
                    format->devicePtr._meta,
                    format->devicePtr._indices,
                    B_dense.device_ptr,
                    C.device_ptr, D_trans.device_ptr);

            // HorizontalSpmmSparseTransKernelExec<BlockShape, WarpShape, MmaShape, 2>(
            //         m, dense_B_lens, k, vector_length, N, M,
            //         format->devicePtr._value,
            //         format->devicePtr._meta,
            //         format->devicePtr._indices,
            //         B_dense.device_ptr, B.indices_device_ptr, B.indices_len, routing_weights.device_ptr, C.device_ptr, D_trans.device_ptr);
        }
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