set(GEMM_SOURCES
        cublas_gemm.h
        cublasLt_gemm.h
        cublas_gemm.cpp
        cublasLt_gemm.cpp
)
set_source_files_properties(${GEMM_SOURCES} PROPERTIES LANGUAGE CUDA)

add_library(gemm STATIC
        ${GEMM_SOURCES})