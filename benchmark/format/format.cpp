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


#include <random>
#include "format.hpp"


template<typename T>
Format<T>::Format(uint seed)
        : rows(0), cols(0), seed(seed) {}

template<typename T>
Format<T>::Format(int rows, int cols, uint seed): rows(rows), cols(cols), seed(seed) {}

template<typename T>
Format<T>::~Format() = default;

template<typename T>
void Format<T>::init() {
    create_data_randomly();
}

template<typename T>
void Format<T>::create_data_randomly() {
    this->data.resize(this->cols * this->rows);

    for (int iter_row = 0; iter_row < rows; ++iter_row) {
        for (int iter_col = 0; iter_col < cols; ++iter_col) {
            data[iter_row * cols + iter_col] = static_cast<T>((float) (std::rand() % 9 - 4));
        }
    }
}

template<typename T>
void Format<T>::create_data_ordered() {
    this->data.resize(cols * rows);
    for (int iter_row = 0; iter_row < rows; ++iter_row) {
        for (int iter_col = 0; iter_col < cols; ++iter_col) {
                data[iter_row * cols + iter_col] = iter_row * cols + iter_col;
        }
    }
}

template<typename T>
void Format<T>::create_data_ordered_interveled() {
    this->data.resize(cols * rows);
    for (int iter_row = 0; iter_row < rows; ++iter_row) {
        for (int iter_col = 0; iter_col < cols; ++iter_col) {
            int a;
            if (iter_col % 8 >= 2 || iter_col >= 64 || (iter_row * cols + iter_col) / 4096 >= 16)
                a = 0;
            else
                a = iter_col / 8 * 2 + (((iter_col % 8) / 4) ? 0 : iter_col % 8) + iter_row * 16;
            this->data[iter_row * cols + iter_col] = static_cast<T>(a);
        }
    }
}

template<typename T>
void Format<T>::create_data_ordered_small() {
    this->data.resize(this->cols * this->rows);
    int temp = 0;
    for (int iter_row = 0; iter_row < this->rows; ++iter_row) {
        for (int iter_col = 0; iter_col < cols; ++iter_col) {
            if (iter_row % 4 >= 2 || iter_row >= 8 || iter_col % 4 >= 2 || iter_col >= 8) {
                this->data[iter_row * cols + iter_col] = static_cast<T>(0);
            } else {
                this->data[iter_row * cols + iter_col] = static_cast<T>(temp);
                // temp += 1;
                temp = (temp + 1) % 128;
            }
        }
    }
}

template
class Format<__half>;