cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

project(flexible_spmm CUDA CXX C)

include(CMakePrintHelpers)
set(CUDA_TOOLKIT_PATH "/usr/local/cuda/")

set(CMAKE_CUDA_FLAGS "-m64 -lineinfo")

# TODO:(CMake Flags for Debug)
#set(CMAKE_CUDA_FLAGS "-G")
#set(CUDA_NVCC_FLAGS -G;-g)

add_subdirectory(benchmark)
