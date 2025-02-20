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


#ifndef FLEXIBLE_SPMM_DATA_STRUCTURE_HELPER_H
#define FLEXIBLE_SPMM_DATA_STRUCTURE_HELPER_H

#include <cuda_fp16.h>
#include <stdio.h>

#include "../mm_utils/spmm_utils.h"

typedef ShapeBase<16, 32, 8> Shape_16_32_8;

template<typename Shape>
struct A_fragment;
template<typename Shape>
struct B_fragment;
template<typename Shape, typename AccumulatorType>
struct C_fragment;
template<typename Shape>
struct Meta_fragment;
template<>
struct A_fragment<Shape_16_32_8> {
    uint x[4];
};
template<>
struct B_fragment<Shape_16_32_8> {
    uint x[4];
};
template<>
struct C_fragment<Shape_16_32_8, float> {
    float x_p1[2] = {0};
    float x_p2[2] = {0};
};
template<>
struct C_fragment<Shape_16_32_8, half> {
    uint x_p1[1] = {0};
    uint x_p2[1] = {0};
};
template<>
struct Meta_fragment<Shape_16_32_8> {
    uint x[1];
};

#endif //FLEXIBLE_SPMM_DATA_STRUCTURE_HELPER_H
