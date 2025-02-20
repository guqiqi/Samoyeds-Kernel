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


#ifndef FLEXIBLE_SPMM_MMA_HELPER_H
#define FLEXIBLE_SPMM_MMA_HELPER_H

#include "data_structure_helper.h"
// __device__ __forceinline__
// void mma_sync_sparse(
//         C_fragment<Shape_16_32_8, float> &d,
//         const A_fragment<Shape_16_32_8> &a,
//         const B_fragment<Shape_16_32_8> &b,
//         const C_fragment<Shape_16_32_8, float> &c,
//         const Meta_fragment<Shape_16_32_8> &e) {
//     asm volatile(
//             "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
//             "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10,%11}, "
//             "{%12,%13,%14,%15}, %16, 0x0;\n"
//             :"=f"(d.x[0]), "=f"(d.x[1]), "=f"(d.x[2]), "=f"(d.x[3])
//             : "r"(a.x[0]), "r"(a.x[2]), "r"(a.x[1]), "r"(a.x[3]),
//     "r"(b.x[0]), "r"(b.x[1]), "r"(b.x[2]), "r"(b.x[3]),
//     "f"(c.x[0]), "f"(c.x[1]), "f"(c.x[2]), "f"(c.x[3]), "r"(e.x[0])
//             );
// }

__device__ __forceinline__
void mma_sync_sparse(
        C_fragment<Shape_16_32_8, float> &d1,
        C_fragment<Shape_16_32_8, float> &d2,
        const A_fragment<Shape_16_32_8> &a,
        const B_fragment<Shape_16_32_8> &b,
        const C_fragment<Shape_16_32_8, float> &c1,
        const C_fragment<Shape_16_32_8, float> &c2,
        const Meta_fragment<Shape_16_32_8> &e) {
    asm volatile(
            "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10,%11}, "
            "{%12,%13,%14,%15}, %16, 0x0;\n"
            :"=f"(d1.x_p1[0]), "=f"(d1.x_p1[1]), "=f"(d2.x_p2[0]), "=f"(d2.x_p2[1])
            : "r"(a.x[0]), "r"(a.x[2]), "r"(a.x[1]), "r"(a.x[3]),
    "r"(b.x[0]), "r"(b.x[1]), "r"(b.x[2]), "r"(b.x[3]),
    "f"(c1.x_p1[0]), "f"(c1.x_p1[1]), "f"(c2.x_p2[0]), "f"(c2.x_p2[1]), "r"(e.x[0])
            );
}


__device__ __forceinline__
void mma_sync_sparse(
        C_fragment<Shape_16_32_8, float> &d,
        const A_fragment<Shape_16_32_8> &a,
        const B_fragment<Shape_16_32_8> &b,
        const C_fragment<Shape_16_32_8, float> &c,
        const Meta_fragment<Shape_16_32_8> &e) {
    asm volatile(
            "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10,%11}, "
            "{%12,%13,%14,%15}, %16, 0x0;\n"
            :"=f"(d.x_p1[0]), "=f"(d.x_p1[1]), "=f"(d.x_p2[0]), "=f"(d.x_p2[1])
            : "r"(a.x[0]), "r"(a.x[2]), "r"(a.x[1]), "r"(a.x[3]),
    "r"(b.x[0]), "r"(b.x[1]), "r"(b.x[2]), "r"(b.x[3]),
    "f"(c.x_p1[0]), "f"(c.x_p1[1]), "f"(c.x_p2[0]), "f"(c.x_p2[1]), "r"(e.x[0])
            );
}

__device__ __forceinline__
void mma_sync_sparse(
        C_fragment<Shape_16_32_8, half> &d,
        const A_fragment<Shape_16_32_8> &a,
        const B_fragment<Shape_16_32_8> &b,
        const C_fragment<Shape_16_32_8, half> &c,
        const Meta_fragment<Shape_16_32_8> &e) {
    asm volatile(
            "mma.sp.sync.aligned.m16n8k32.row.col.f16.f16.f16.f16 "
            "{%0, %1}, {%2, %3, %4, %5}, {%6, %7, %8, %9}, "
            "{%10,%11}, %12, 0x0;\n"
            : "=r"(d.x_p1[0]), "=r"(d.x_p2[0])
            : "r"(a.x[0]), "r"(a.x[2]), "r"(a.x[1]), "r"(a.x[3]),
            "r"(b.x[0]), "r"(b.x[1]), "r"(b.x[2]), "r"(b.x[3]),
            "r"(c.x_p1[0]), "r"(c.x_p2[0]), "r"(e.x[0])
            );
}

#endif //FLEXIBLE_SPMM_MMA_HELPER_H
