//
// Created by cpwu on 2/28/24.
//

#include "format_ss.hpp"

#include <random>

#include "../util/utils.h"

template<typename T>
Format_ss<T>::Format_ss(uint seed) : Format<T>(seed) {}

template<typename T>
Format_ss<T>::Format_ss(int rows, int cols, uint seed, int vector_length, int structure_N, int structure_M) : Format<T>(
        rows, cols, seed), vector_length(vector_length), structure_N(structure_N), structure_M(structure_M) {}

template<typename T>
Format_ss<T>::~Format_ss() = default;

template<typename T>
void Format_ss<T>::init() {
    Format<T>::create_data_randomly();
    // Format<T>::create_data_ordered_small();
    // Format<T>::create_data_ordered();
    // create_indices_ordered();
    // create_metadata_ordered();
    create_indices_randomly();
    // create_metadata_randomly();
    // 这一步创建的是metadata_uncompressed_uint的数据
    create_metadata_uint_randomly();
    // 这一步将根据metadata_uncompressed_uint创建metadata数据，这个数据是以short形式读取顺序存储的
    // 注意，这部分不可被删除，因为legacy code的原因，下面的pruned value是根据metadata的内容裁剪的
    convert_metadata_uint22bit_short_load_format();
    // 这一步将根据metadata_uncompressed_uint创建metadata_uint_load数据,这个数据是以uint形式读取顺序存储的
    convert_metadata_uint22bit_uint_load_format();
    create_value();
    create_pruned_value();
    transpose_indices();
}

template<typename T>
void Format_ss<T>::transpose_indices() {
    std::vector<uint> temp;
    temp.resize(get_num_of_rows_indices() * get_num_of_cols_indices());
    for(int iter_cols = 0; iter_cols < get_num_of_cols_indices(); iter_cols++) {
        for(int iter_rows = 0; iter_rows < get_num_of_rows_indices(); iter_rows++) {
            temp[iter_cols * get_num_of_rows_indices() + iter_rows] = this->indices[iter_rows * get_num_of_cols_indices() + iter_cols];
        }
    }
    this->indices = temp;
}

//TODO: (CPWU) 这里的indices是应该行优先还是列优先？暂时按照与value一致的方式
template<typename T>
void Format_ss<T>::create_indices_ordered() {
    this->indices.resize(get_num_of_rows_indices() * get_num_of_cols_indices());
    for(int iter_cols = 0; iter_cols < get_num_of_cols_indices(); iter_cols++) {
        for(int iter_rows = 0; iter_rows < get_num_of_rows_indices(); iter_rows += structure_N) {
            for (int i = 0; i < structure_N; i++) {
                this->indices[(iter_rows + i) * get_num_of_cols_indices() + iter_cols] = i;
            }
        }
    }
}

template<typename T>
void Format_ss<T>::create_indices_randomly() {
    std::vector<uint> temp;
    temp.reserve(structure_M);
    for (int i = 0; i < structure_M; i++) {
        temp.push_back(i);
    }
    this->indices.resize(get_num_of_rows_indices() * get_num_of_cols_indices());
    for(int iter_cols = 0; iter_cols < get_num_of_cols_indices(); iter_cols++) {
        for(int iter_rows = 0; iter_rows < get_num_of_rows_indices(); iter_rows += structure_N) {
            random_shuffle(temp.begin(), temp.end());
            std::sort(temp.begin(), temp.begin() + structure_N);
            for (int i = 0; i < structure_N; i++) {
                this->indices[(iter_rows + i) * get_num_of_cols_indices() + iter_cols] = temp[i];
            }
        }
    }
}

template<typename T>
void Format_ss<T>::create_metadata_ordered() {
    this->metadata.resize(get_num_of_rows_metadata() * get_num_of_cols_metadata());
    for (int iter_rows = 0; iter_rows < get_num_of_rows_metadata(); ++iter_rows) {
        for (int iter_cols = 0; iter_cols < get_num_of_cols_metadata(); ++iter_cols) {
            uint temp_meta = 0;
            for (int i = 0; i < NUM_OF_META_PER_UINT; i += SPTC_N) {
                for (int j = 0; j < SPTC_N; ++j) {
                    temp_meta = temp_meta | j << ((i + j) * METADATA_SIZE);
                }
            }
            this->metadata[iter_rows * get_num_of_cols_metadata() + iter_cols] = temp_meta;
        }
    }
}

template<typename T>
void Format_ss<T>::create_metadata_randomly() {
    std::vector<uint> temp;
    temp.reserve(SPTC_M);
    for (int i = 0; i < SPTC_M; i++) {
        temp.push_back(i);
    }
    this->metadata.resize(get_num_of_rows_metadata() * get_num_of_cols_metadata());
    for (int iter_rows = 0; iter_rows < get_num_of_rows_metadata(); ++iter_rows) {
        for (int iter_cols = 0; iter_cols < get_num_of_cols_metadata(); ++iter_cols) {
            uint temp_meta = 0;
            for (int i = 0; i < NUM_OF_META_PER_UINT; i += SPTC_N) {
                random_shuffle(temp.begin(), temp.end());
                std::sort(temp.begin(), temp.begin() + SPTC_N);
                for (int j = 0; j < SPTC_N; ++j) {
                    temp_meta = temp_meta | temp[j] << ((i + j) * METADATA_SIZE);
                }
            }
            this->metadata[iter_rows * get_num_of_cols_metadata() + iter_cols] = temp_meta;
        }
    }
}

template<typename T>
void Format_ss<T>::create_metadata_uint_ordered() {
    // metadata没有压缩为2-bit前的元素个数应当与value的元素数量相同
    this->metadata_uncompressed_uint.resize(get_num_of_rows_value() * get_num_of_cols_value());
    for (int iter_rows = 0; iter_rows < get_num_of_rows_value(); ++iter_rows) {
        for (int iter_cols = 0; iter_cols < get_num_of_cols_value(); iter_cols += SPTC_N) {
            for (int i = 0; i < SPTC_N; ++i) {
                metadata_uncompressed_uint[iter_rows * get_num_of_cols_value() + iter_cols + i] = i;
            }
        }
    }
}

template<typename T>
void Format_ss<T>::create_metadata_uint_randomly() {
    std::vector<uint> temp;
    temp.reserve(SPTC_M);
    for (int i = 0; i < SPTC_M; i++) {
        temp.push_back(i);
    }
    // metadata没有压缩为2-bit前的元素个数应当与value的元素数量相同
    this->metadata_uncompressed_uint.resize(get_num_of_rows_value() * get_num_of_cols_value());
    for (int iter_rows = 0; iter_rows < get_num_of_rows_value(); ++iter_rows) {
        for (int iter_cols = 0; iter_cols < get_num_of_cols_value(); iter_cols += SPTC_N) {
            random_shuffle(temp.begin(), temp.end());
            std::sort(temp.begin(), temp.begin() + SPTC_N);
            for (int i = 0; i < SPTC_N; ++i) {
                metadata_uncompressed_uint[iter_rows * get_num_of_cols_value() + iter_cols + i] = temp[i];
            }
        }
    }
}

template<typename T>
void Format_ss<T>::convert_metadata_uint22bit_short_load_format() {
    this->metadata.resize(get_num_of_rows_metadata() * get_num_of_cols_metadata());
    for (int iter_rows = 0; iter_rows < get_num_of_rows_metadata(); ++iter_rows) {
        for (int iter_cols = 0; iter_cols < get_num_of_cols_metadata(); ++iter_cols) {
            uint temp_meta = 0;
            for (int i = 0; i < NUM_OF_META_PER_UINT; i++) {
                temp_meta = temp_meta |
                    (this->metadata_uncompressed_uint[iter_rows * get_num_of_cols_value() +
                    iter_cols * NUM_OF_META_PER_UINT + i] << (i * METADATA_SIZE));
            }
            this->metadata[iter_rows * get_num_of_cols_metadata() + iter_cols] = temp_meta;
        }
    }
}

// 交换 16x16 子矩阵的右上角和左下角的 8x8 子矩阵
void swapSubMatrices(std::vector<uint>& matrix, int offset, int num_cols) {
    int unit_rows = 16;
    int unit_cols = 16;
    int half = unit_rows / 2; // 8

    for (int i = 0; i < half; ++i) {
        for (int j = half; j < unit_cols; ++j) {
            // 计算右上角元素的索引
            int top_right_index = offset + i * num_cols + j;
            // 计算左下角元素的索引
            int bottom_left_index = offset + (i + half) * num_cols + (j - half);
            // 交换两个元素
            std::swap(matrix[top_right_index], matrix[bottom_left_index]);
        }
    }
}

// TODO: 这里的16 * 16和上面的swap方法是耦合的
template<typename T>
void Format_ss<T>::convert_metadata_uint22bit_uint_load_format() {
    // 以16 row * (1 uint或者说 16 2-bit)为单位进行数据变换
    int unit_rows = 16;
    int unit_cols = 16;
    // std::vector<uint> temp(unit_rows * unit_cols);
    for (int iter_rows = 0; iter_rows < get_num_of_rows_value() / unit_rows; ++iter_rows) {
        for (int iter_cols = 0; iter_cols < get_num_of_cols_value() / unit_cols; ++iter_cols) {
            // temp.assign(temp.size(), 0);
            // 矩阵左上角的位置
            int offset = iter_rows * unit_rows * get_num_of_cols_value() + iter_cols * unit_cols;
            swapSubMatrices(this->metadata_uncompressed_uint, offset, get_num_of_cols_value());
        }
    }

    this->metadata_uint_load.resize(get_num_of_rows_metadata() * get_num_of_cols_metadata());
    for (int iter_rows = 0; iter_rows < get_num_of_rows_metadata(); ++iter_rows) {
        for (int iter_cols = 0; iter_cols < get_num_of_cols_metadata(); ++iter_cols) {
            uint temp_meta = 0;
            for (int i = 0; i < NUM_OF_META_PER_UINT; i++) {
                temp_meta = temp_meta |
                            (this->metadata_uncompressed_uint[iter_rows * get_num_of_cols_value() +
                                                              iter_cols * NUM_OF_META_PER_UINT + i] << (i * METADATA_SIZE));
            }
            this->metadata_uint_load[iter_rows * get_num_of_cols_metadata() + iter_cols] = temp_meta;
        }
    }
}

template<typename T>
void Format_ss<T>::create_value() {
    this->value.resize(get_num_of_rows_value() * get_num_of_cols_value());
    for (int iter_row = 0; iter_row < get_num_of_rows_value(); ++iter_row) {
        for (int iter_col = 0; iter_col < get_num_of_cols_value(); iter_col += SPTC_N) {
            int indices_row_idx = iter_row;
            int indices_col_idx = iter_col / (vector_length / SPTC_M * SPTC_N);
            int selected_index = this->indices[indices_row_idx * get_num_of_cols_indices() + indices_col_idx];
            int row_offset = iter_row / structure_N * structure_M + selected_index;

            int metadata_row_idx = iter_row;
            int metadata_col_idx = iter_col / NUM_OF_META_PER_UINT;
            int metadata_inside_idx =
                    iter_col % NUM_OF_META_PER_UINT; //每个uint包含NUM_OF_META_PER_UINT(16)个metadata，计算需要取到第几位
            int mask = ((0b1 << (SPTC_N * METADATA_SIZE)) - 1) << (metadata_inside_idx * METADATA_SIZE);
            uint selected_metadata = (metadata[metadata_row_idx * get_num_of_cols_metadata() + metadata_col_idx] & mask)
                    >> (metadata_inside_idx * METADATA_SIZE);
            int selected_metadata_entry[SPTC_N];
            int temp = static_cast<int>(selected_metadata);
            int divider = 0b1 << METADATA_SIZE;
            for (int i = 0; i < SPTC_N; ++i) {
                selected_metadata_entry[i] = temp % divider;
                temp /= divider;
            }
            for (int i = 0; i < SPTC_N; ++i) {
                int des = iter_row * get_num_of_cols_value() + iter_col + i;
                int index = row_offset * this->cols + iter_col / SPTC_N * SPTC_M + selected_metadata_entry[i];
                this->value[iter_row * get_num_of_cols_value() + iter_col + i] =
                        this->data[index];
            }
        }
    }
}

template<typename T>
void Format_ss<T>::create_pruned_value() {
    pruned_value.resize(get_num_of_rows_pruned_value() * get_num_of_cols_pruned_value());
    for (int iter_row = 0; iter_row < get_num_of_rows_value(); ++iter_row) {
        for (int iter_col = 0; iter_col < get_num_of_cols_value(); iter_col += structure_N) {
            pruned_value[iter_row * get_num_of_cols_pruned_value() + iter_col] = 0;
        }
    }

    for (int iter_row = 0; iter_row < get_num_of_rows_value(); ++iter_row) {
        for (int iter_col = 0; iter_col < get_num_of_cols_value(); iter_col += SPTC_N) {
            int indices_row_idx = iter_row;
            int indices_col_idx = iter_col / (vector_length / SPTC_M * SPTC_N);
            int selected_index = this->indices[indices_row_idx * get_num_of_cols_indices() + indices_col_idx];
            int row_offset = iter_row / structure_N * structure_M + selected_index;

            int metadata_row_idx = iter_row;
            int metadata_col_idx = iter_col / NUM_OF_META_PER_UINT;
            int metadata_inside_idx =
                    iter_col % NUM_OF_META_PER_UINT; //每个uint包含NUM_OF_META_PER_UINT(16)个metadata，计算需要取到第几位
            int mask = ((0b1 << (SPTC_N * METADATA_SIZE)) - 1) << (metadata_inside_idx * METADATA_SIZE);
            uint selected_metadata = (metadata[metadata_row_idx * get_num_of_cols_metadata() + metadata_col_idx] & mask)
                    >> (metadata_inside_idx * METADATA_SIZE);
            int selected_metadata_entry[SPTC_N];
            int temp = static_cast<int>(selected_metadata);
            int divider = 0b1 << METADATA_SIZE;
            for (int i = 0; i < SPTC_N; ++i) {
                selected_metadata_entry[i] = temp % divider;
                temp /= divider;
            }
            for (int i = 0; i < SPTC_N; ++i) {
                int index = row_offset * this->cols + iter_col / SPTC_N * SPTC_M + selected_metadata_entry[i];
                this->pruned_value[index] = this->data[index];
            }
        }
    }
}

template<typename T>
void Format_ss<T>::sync_device() {
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
    // CUDA_CHECK(cudaMemcpy(meta_ptr, metadata.data(), meta_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(meta_ptr, metadata_uint_load.data(), meta_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(value_ptr, value.data(), value_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pruned_value_ptr, pruned_value.data(), pruned_value_size, cudaMemcpyHostToDevice));

    devicePtr._indices = indices_ptr;
    devicePtr._meta = meta_ptr;
    devicePtr._value = value_ptr;
    devicePtr._pruned_value = pruned_value_ptr;
}

template
class Format_ss<__half>;