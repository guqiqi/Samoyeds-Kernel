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


#ifndef FLEXIBLE_SPMM_MEMCPY_UTIL_H
#define FLEXIBLE_SPMM_MEMCPY_UTIL_H

__device__ __forceinline__
uint get_smem_ptr(const void *ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

#endif //FLEXIBLE_SPMM_MEMCPY_UTIL_H
