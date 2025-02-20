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

#ifndef FLEXIBLE_SPMM_RESULT_CHECK_H
#define FLEXIBLE_SPMM_RESULT_CHECK_H

#include <iostream>

template <typename T>
bool check_results(std::vector<T> cusparse_result, std::vector<T> cuBLAS_result, int row, int col) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            float c_value = static_cast<float>(cusparse_result[i * col + j]);
            float c_result = static_cast<float>(cuBLAS_result[i * col + j]);
            if (abs(c_value - c_result) > 1.0 && (c_value / c_result > 1.0001 || c_result / c_value > 1.0001 || abs(c_value - c_result) > 1e-5))
            {
                std::cout << "error: " << i << " " << j << " " << c_value << " " << c_result << std::endl;
                return false;
            }
        }
    }
    return true;
}

#endif //FLEXIBLE_SPMM_RESULT_CHECK_H
