//
// Created by cpwu on 2/28/24.
//

#ifndef FLEXIBLE_SPMM_FORMAT_SS_HPP
#define FLEXIBLE_SPMM_FORMAT_SS_HPP

#include "format.hpp"

#define METADATA_SIZE 2  // 每个metadata占两个bit
#define BITS_PER_BYTE 8  // 1 Byte = 8 Bits
#define NUM_OF_META_PER_UINT static_cast<int>(sizeof(uint) * BITS_PER_BYTE / METADATA_SIZE)
#define SPTC_N 2    // NVIDIA SPTC (Sparse Tensor Core)
#define SPTC_M 4    // NVIDIA SPTC (Sparse Tensor Core)

template<typename T>
class Format_ss : public Format<T> {
public:
    Format_ss(uint seed);

    // Choose N rows out of every M rows, every vector_length cols share the same selection
    Format_ss(int rows, int cols, uint seed, int vector_length, int structure_N, int structure_M);

    ~Format_ss();

    void init();

    int structure_N, structure_M; // Choose N rows out of every M rows
    int vector_length; // every vector_length cols share the same selection

    std::vector<uint> indices;
    std::vector<uint> metadata_uncompressed_uint; // 用于存储metadata在被压缩为2-bit内存占用前的数据
    std::vector<uint> metadata;
    std::vector<uint> metadata_uint_load;
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
        return this->rows / structure_M * structure_N;
    }

    int get_num_of_cols_indices() {
        return this->cols / vector_length;
    }

    int get_num_of_rows_metadata() {
        return get_num_of_rows_value();
    }

    int get_num_of_cols_metadata() {
        return get_num_of_cols_value() / NUM_OF_META_PER_UINT;
    }

    int get_num_of_rows_value() {
        return this->rows / structure_M * structure_N;
    }

    int get_num_of_cols_value() {
        return this->cols / SPTC_M * SPTC_N;
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

    void create_metadata_uint_ordered();
    void create_metadata_uint_randomly();
    void convert_metadata_uint22bit_short_load_format();
    void convert_metadata_uint22bit_uint_load_format();

    void create_value();

    void create_pruned_value();

    void transpose_indices();

    void sync_device();
};

#endif //FLEXIBLE_SPMM_FORMAT_SS_HPP
