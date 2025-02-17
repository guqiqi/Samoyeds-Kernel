//
// Created by kiki on 24-3-6.
//

#ifndef FLEXIBLE_SPMM_MEMCPY_UTIL_H
#define FLEXIBLE_SPMM_MEMCPY_UTIL_H

__device__ __forceinline__
uint get_smem_ptr(const void *ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

#endif //FLEXIBLE_SPMM_MEMCPY_UTIL_H
