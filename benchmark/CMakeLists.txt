add_subdirectory(util)
add_subdirectory(mm_utils)
add_subdirectory(format)
add_subdirectory(gemm)
add_subdirectory(ssmm)

set(METHODS_HEADERS
        methods/ssmm_timer.h
        methods/cublas_timer.h
        methods/cublasLt_timer.h
)

set(FORMAT_CHECK_SOURCES
        format_check.cpp)
set_source_files_properties(${FORMAT_CHECK_SOURCES} PROPERTIES LANGUAGE CUDA)
add_executable(format_check
        ${FORMAT_CHECK_SOURCES})
set_source_files_properties(${format_check} PROPERTIES LANGUAGE CUDA)
target_link_libraries(format_check
        util format)

set(HORIZONTAL_SSMM_KERNEL_BENCH_SOURCES
        horizontal_ssmm_kernel_benchmark.cpp)
set_source_files_properties(${HORIZONTAL_SSMM_KERNEL_BENCH_SOURCES} PROPERTIES LANGUAGE CUDA)
add_executable(horizontal_ssmm_benchmark
        ${HORIZONTAL_SSMM_KERNEL_BENCH_SOURCES})
set_source_files_properties(${horizontal_ssmm_benchmark} PROPERTIES LANGUAGE CUDA)
target_link_libraries(horizontal_ssmm_benchmark
        util format ssmm gemm cublasLt cublas)

set(HORIZONTAL_SPMM_KERNEL_BENCH_SOURCES
        horizontal_spmm_kernel_benchmark.cpp)
set_source_files_properties(${HORIZONTAL_SPMM_KERNEL_BENCH_SOURCES} PROPERTIES LANGUAGE CUDA)
add_executable(horizontal_spmm_benchmark
        ${HORIZONTAL_SPMM_KERNEL_BENCH_SOURCES})
set_source_files_properties(${horizontal_spmm_benchmark} PROPERTIES LANGUAGE CUDA)
target_link_libraries(horizontal_spmm_benchmark
        util format ssmm gemm cublasLt cublas)

set(BENCHMARK_SOURCES
        benchmark.cpp)
set_source_files_properties(${BENCHMARK_SOURCES} PROPERTIES LANGUAGE CUDA)
add_executable(benchmark
        ${BENCHMARK_SOURCES} ${METHODS_HEADERS})
set_source_files_properties(${benchmark} PROPERTIES LANGUAGE CUDA)
target_link_libraries(benchmark
        util format ssmm gemm cublasLt cublas)
