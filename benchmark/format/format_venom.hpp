//
// Created by CPWu2017@126.com on 12/22/23.
//

#ifndef FLEXIBLE_SPMM_FORMAT_VENOM_HPP
#define FLEXIBLE_SPMM_FORMAT_VENOM_HPP

#include "format.hpp"

#define METADATA_SIZE 2  // 每个metadata占两个bit
#define BITS_PER_BYTE 8  // 1 Byte = 8 Bits
#define NUM_OF_META_PER_UINT static_cast<int>(sizeof(uint) * BITS_PER_BYTE / METADATA_SIZE)

template<typename T>
class Format_venom : public Format<T> {
public:
    Format_venom(uint seed);

    Format_venom(int rows, int cols, uint seed, int structure_V, int structure_N, int structure_M, int vector_length);

    ~Format_venom();

    void init();

    int structure_V, structure_N, structure_M; //V:N:M 8:2:4
    int vector_length; // vector-wise方向上的长度

    std::vector<uint> indices;
    std::vector<uint> metadata;
    std::vector<T> value; // SPMM所需数据，稀疏位置跳过
    std::vector<T> pruned_value; // GEMM所需剪裁数据，稀疏位置补0

    struct Device_ptr {
        uint *_indices = nullptr;
        uint *_meta = nullptr;
        T *_value = nullptr;
        T *_pruned_value = nullptr;
    };
    Device_ptr devicePtr;

    int get_num_of_rows_indices() {
        // OPTIMIZE: (CPWU) 需要针对无法整除的情况优化
        return this->rows / vector_length;
    }
    int get_num_of_cols_indices() {
        // OPTIMIZE: (CPWU) 需要针对无法整除的情况优化
        return this->cols / structure_V * structure_M;
    }

    int get_num_of_rows_metadata() {
        return this->rows;
    }
    int get_num_of_cols_metadata() {
        return get_num_of_cols_value() / NUM_OF_META_PER_UINT;
    }

    int get_num_of_rows_value() {
        return this->rows;
    }
    int get_num_of_cols_value() {
        // OPTIMIZE: (CPWU) 需要针对无法整除的情况优化
        return this->cols / structure_V * structure_N;
    }

    int get_num_of_rows_pruned_value() {
        return this->rows;
    }
    int get_num_of_cols_pruned_value() {
        return this->cols;
    }

    void create_indices_randomly();
    void create_indices_ordered();

    void create_metadata_randomly();
    void create_metadata_ordered();

    void create_value();

    void create_pruned_value();

    void sync_device();
};


#endif //FLEXIBLE_SPMM_FORMAT_VENOM_HPP
