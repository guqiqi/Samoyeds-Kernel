//
// Created by cpwu on 2/21/24.
//

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
