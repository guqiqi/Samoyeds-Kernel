//
// Created by cpwu on 24-4-9.
//
#include <iostream>
#include <chrono>
#include <random>
#include "./util/argparse.hpp"

#include "./methods/ssmm_timer.h"
#include "./methods/cublas_timer.h"
// #include "./methods/cublasLt_timer.h"

int main(int argc, const char **argv) {
    argparse::ArgumentParser parser("benchmark for matrix multiplication implementations.");
    parser.add_argument("-m").help("Rows number of A").default_value(1024).scan<'i', int>();
    parser.add_argument("-k").help("Cols number of A").default_value(1024).scan<'i', int>();
    parser.add_argument("-n").help("Cols number of B").default_value(16384).scan<'i', int>();
    parser.add_argument("-V").help("V of structural sparsity V:N:M").default_value(8).scan<'i', int>();
    parser.add_argument("-N").help("N of structural sparsity V:N:M").default_value(1).scan<'i', int>();
    parser.add_argument("-M").help("M of structural sparsity V:N:M").default_value(2).scan<'i', int>();
    parser.add_argument("-d", "--density").help("Density of A").default_value(0.5).scan<'g', double>();
    parser.add_argument("--vector_length").help("vector-wise方向上的长度").default_value(128).scan<'i', int>();
    parser.add_argument("-s", "--seed").help("Set a random seed, or it is created randomly").scan<'i', int>();
    parser.add_argument("--method").help(
            "matrix multiplication method. Can be [cuBlas, cuSparseLt, Spatha, Sputnik, SSMM]").default_value("SSMM");

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
        seed = parser.get<int>("--seed");
    else
        seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::srand(seed);

    auto method = parser.get<std::string>("--method");
    if (method == "SSMM") {
        SSMM_timer<half>(m, k, n, N, M, vector_length, seed);
    } else if (method == "cuBlas") {
        cuBlas_timer<half>(m, k, n, N, M, vector_length, seed);
    } else if (method == "cuBlasLt") {
        std::cerr << "Method not supported" << std::endl;
        return 1;
    } else if (method == "cuSparseLt") {
        std::cerr << "Method not supported" << std::endl;
        return 1;
    } else if (method == "Spatha") {
        std::cerr << "Method not supported" << std::endl;
        return 1;
    } else {
        std::cerr << "Method not supported" << std::endl;
        return 1;
    }
    return 0;
}