//
// Created by cpwu on 1/3/24.
//

#ifndef FLEXIBLE_SPMM_UTILS_H
#define FLEXIBLE_SPMM_UTILS_H

#include <iostream>
#include <random>

#define CUDA_CHECK(status)                                              \
  {                                                                     \
    cudaError_t error = status;                                         \
    if (error != cudaSuccess) {                                         \
      printf("*** CUDA Error *** at [%s:%d] error=%d, reason:%s \n",    \
            __FILE__, __LINE__, error, cudaGetErrorString(error));      \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
}

#endif //FLEXIBLE_SPMM_UTILS_H
