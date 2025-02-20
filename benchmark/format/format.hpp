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


#ifndef FLEXIBLE_SPMM_FORMAT_HPP
#define FLEXIBLE_SPMM_FORMAT_HPP

#include <vector>
#include <iostream>
#include <cuda_fp16.h>
#include <algorithm>

template<typename T>
class Format {
public:
    Format(uint seed);
    Format(int rows, int cols, uint seed);
    ~Format();

    virtual void init();
    void create_data_randomly();
    void create_data_ordered();
    void create_data_ordered_interveled();
    void create_data_ordered_small();

    virtual void sync_device() = 0;

    std::vector<T> data; // Origin data
    int rows, cols;
    uint seed;
};

#endif //FLEXIBLE_SPMM_FORMAT_HPP
