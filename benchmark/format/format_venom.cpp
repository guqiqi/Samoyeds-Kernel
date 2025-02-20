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


#include "format_venom.hpp"
#include "../util/utils.h"

template<typename T>
Format_venom<T>::Format_venom(uint seed) : Format<T>(seed) {}

template<typename T>
Format_venom<T>::Format_venom(int rows, int cols, uint seed, int structure_V, int structure_N, int structure_M,
                              int vector_length)
        : Format<T>(rows, cols, seed), structure_V(structure_V), structure_N(structure_N), structure_M(structure_M),
          vector_length(vector_length) {}

template<typename T>
Format_venom<T>::~Format_venom() = default;

template<typename T>
void Format_venom<T>::init() {
    Format<T>::create_data_randomly();
    // Format<T>::create_data_ordered();
    // Format<T>::create_data_ordered_interveled();
    // Format<T>::create_data_ordered_small();
    // create_indices_ordered();
    // create_metadata_ordered();
    create_indices_randomly();
    create_metadata_randomly();
    create_value();
    create_pruned_value();
}

template<typename T>
void Format_venom<T>::create_indices_randomly() {
    std::vector<uint> temp;
    temp.reserve(structure_V);
    for (int i = 0; i < structure_V; ++i) {
        temp.push_back(i);
    }
    this->indices.resize(get_num_of_rows_indices() * get_num_of_cols_indices());
    // OPTIMIZE: (CPWU) 需要针对无法整除的情况优化
    for (int iter_rows = 0; iter_rows < get_num_of_rows_indices(); ++iter_rows) {
        for (int iter_cols = 0; iter_cols < get_num_of_cols_indices(); iter_cols += structure_M) {
            random_shuffle(temp.begin(), temp.end());
            std::sort(temp.begin(), temp.begin() + structure_M);
            for (int i = 0; i < structure_M; ++i) {
                indices[iter_rows * get_num_of_cols_indices() + iter_cols + i] = temp[i];
            }
        }
    }
}

template<typename T>
void Format_venom<T>::create_indices_ordered() {
    this->indices.resize(get_num_of_rows_indices() * get_num_of_cols_indices());
    // OPTIMIZE: (CPWU) 需要针对无法整除的情况优化
    for (int iter_rows = 0; iter_rows < get_num_of_rows_indices(); ++iter_rows) {
        for (int iter_cols = 0; iter_cols < get_num_of_cols_indices(); iter_cols += structure_M) {
            for (int i = 0; i < structure_M; ++i) {
                indices[iter_rows * get_num_of_cols_indices() + iter_cols + i] = i;
            }
        }
    }
}

template<typename T>
void Format_venom<T>::create_metadata_randomly() {
    std::vector<uint> temp;
    temp.reserve(structure_M);
    for (int i = 0; i < structure_M; ++i) {
        temp.push_back(i);
    }
    this->metadata.resize(get_num_of_rows_metadata() * get_num_of_cols_metadata());
    for (int iter_rows = 0; iter_rows < get_num_of_rows_metadata(); ++iter_rows) {
        for (int iter_cols = 0; iter_cols < get_num_of_cols_metadata(); ++iter_cols) {
            uint temp_meta = 0;
            for (int i = 0; i < NUM_OF_META_PER_UINT; i += structure_N) {
                random_shuffle(temp.begin(), temp.end());
                std::sort(temp.begin(), temp.begin() + structure_N);
                for (int j = 0; j < structure_N; ++j) {
                    temp_meta = temp_meta | temp[j] << ((i + j) * METADATA_SIZE);
                }
            }
            this->metadata[iter_rows * get_num_of_cols_metadata() + iter_cols] = temp_meta;
        }
    }
}

template<typename T>
void Format_venom<T>::create_metadata_ordered() {
    this->metadata.resize(get_num_of_rows_metadata() * get_num_of_cols_metadata());
    for (int iter_rows = 0; iter_rows < get_num_of_rows_metadata(); ++iter_rows) {
        for (int iter_cols = 0; iter_cols < get_num_of_cols_metadata(); ++iter_cols) {
            uint temp_meta = 0;
            for (int i = 0; i < NUM_OF_META_PER_UINT; i += structure_N) {
                // NOTE:metadata的 2 : 4 组织应该为每 2 bit 的 small endien
                // 例如，对于32位指示的16个metadata，考虑
                // 对于第1～4列的meta表示取第0列、第1列，应该为0b0100
                // 对于第5～8列的meta表示取第1列、第2列，应该为0b1001
                for (int j = 0; j < structure_N; ++j) {
                    temp_meta = temp_meta | j << ((i + j) * METADATA_SIZE);
                }
            }
            this->metadata[iter_rows * get_num_of_cols_metadata() + iter_cols] = temp_meta;
        }
    }
}

// TODO: (CPWU) 当前只有未swizzle的实现。
template<typename T>
void Format_venom<T>::create_value() {
    this->value.resize(get_num_of_rows_value() * get_num_of_cols_value());
    for (int iter_row = 0; iter_row < get_num_of_rows_value(); ++iter_row) {
        for (int iter_col = 0; iter_col < get_num_of_cols_value(); iter_col += structure_N) {
            // 准备对应的两个indices数据
            int indices_row_idx = iter_row / vector_length;
            int indices_col_idx = iter_col / structure_N * structure_M;
            int start_position = indices_row_idx * get_num_of_cols_indices() + indices_col_idx;
            std::vector<int> selected_indices_entry(indices.begin() + start_position,
                                                    indices.begin() + start_position + structure_M);

            // 准备对应的两个metadata数据
            int metadata_row_idx = iter_row;
            int metadata_col_idx = iter_col / NUM_OF_META_PER_UINT;
            int metadata_inside_idx =
                    iter_col % NUM_OF_META_PER_UINT; //每个uint包含NUM_OF_META_PER_UINT(16)个metadata，计算需要取到第几位
            int mask = ((0b1 << (structure_N * METADATA_SIZE)) - 1) << (metadata_inside_idx * METADATA_SIZE);
            uint selected_metadata = (metadata[metadata_row_idx * get_num_of_cols_metadata() + metadata_col_idx] & mask)
                    >> (metadata_inside_idx * METADATA_SIZE);
            int selected_metadata_entry[structure_N];
            int temp = static_cast<int>(selected_metadata);
            int divider = 0b1 << METADATA_SIZE;
            for (int i = 0; i < structure_N; ++i) {
                selected_metadata_entry[i] = temp % divider;
                temp /= divider;
            }

            // 从data中拿出对应数据到value中
            for (int i = 0; i < structure_N; ++i) {
                value[iter_row * get_num_of_cols_value() + iter_col + i] = this->data[
                        iter_row * this->cols +
                        iter_col / structure_N * structure_V +
                        selected_indices_entry[selected_metadata_entry[i]]];
            }
        }
    }
}

template<typename T>
void Format_venom<T>::create_pruned_value() {
    pruned_value.resize(get_num_of_rows_pruned_value() * get_num_of_cols_pruned_value());

    for (int iter_row = 0; iter_row < get_num_of_rows_value(); ++iter_row) {
        for (int iter_col = 0; iter_col < get_num_of_cols_value(); iter_col += structure_N) {
            pruned_value[iter_row * get_num_of_cols_pruned_value() + iter_col] = 0;
        }
    }
    for (int iter_row = 0; iter_row < get_num_of_rows_value(); ++iter_row) {
        for (int iter_col = 0; iter_col < get_num_of_cols_value(); iter_col += structure_N) {
            // 准备对应的两个indices数据
            int indices_row_idx = iter_row / vector_length;
            int indices_col_idx = iter_col / structure_N * structure_M;
            int start_position = indices_row_idx * get_num_of_cols_indices() + indices_col_idx;
            std::vector<int> selected_indices_entry(indices.begin() + start_position,
                                                    indices.begin() + start_position + structure_M);

            // 准备对应的两个metadata数据
            int metadata_row_idx = iter_row;
            int metadata_col_idx = iter_col / NUM_OF_META_PER_UINT;
            int metadata_inside_idx =
                    iter_col % NUM_OF_META_PER_UINT; //每个uint包含NUM_OF_META_PER_UINT(16)个metadata，计算需要取到第几位
            int mask = ((0b1 << (structure_N * METADATA_SIZE)) - 1) << (metadata_inside_idx * METADATA_SIZE);
            uint selected_metadata = (metadata[metadata_row_idx * get_num_of_cols_metadata() + metadata_col_idx] & mask)
                    >> (metadata_inside_idx * METADATA_SIZE);
            int selected_metadata_entry[structure_N];
            int temp = static_cast<int>(selected_metadata);
            int divider = 0b1 << METADATA_SIZE;
            for (int i = 0; i < structure_N; ++i) {
                selected_metadata_entry[i] = temp % divider;
                temp /= divider;
            }

            // 从data中拿出对应数据到value中
            for (int i = 0; i < structure_N; ++i) {
                int index = iter_row * this->cols +
                            iter_col / structure_N * structure_V +
                            selected_indices_entry[selected_metadata_entry[i]];
                pruned_value[index] = this->data[index];
            }
        }
    }
}

template<typename T>
void Format_venom<T>::sync_device() {
    uint *indices_ptr;
    uint *meta_ptr;
    T *value_ptr;
    T *pruned_value_ptr;

    int indices_size = get_num_of_rows_indices() * get_num_of_cols_indices() * sizeof(uint);
    int meta_size = get_num_of_rows_metadata() * get_num_of_cols_metadata() * sizeof(uint);
    int value_size = get_num_of_rows_value() * get_num_of_cols_value() * sizeof(T);
    int pruned_value_size = get_num_of_rows_pruned_value() * get_num_of_cols_pruned_value() * sizeof(T);

    CUDA_CHECK(cudaMalloc(&indices_ptr, indices_size));
    CUDA_CHECK(cudaMalloc(&meta_ptr, meta_size));
    CUDA_CHECK(cudaMalloc(&value_ptr, value_size));
    CUDA_CHECK(cudaMalloc(&pruned_value_ptr, pruned_value_size));

    CUDA_CHECK(cudaMemcpy(indices_ptr, indices.data(), indices_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(meta_ptr, metadata.data(), meta_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(value_ptr, value.data(), value_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pruned_value_ptr, pruned_value.data(), pruned_value_size, cudaMemcpyHostToDevice));

    devicePtr._indices = indices_ptr;
    devicePtr._meta = meta_ptr;
    devicePtr._value = value_ptr;
    devicePtr._pruned_value = pruned_value_ptr;
}

template
class Format_venom<__half>;