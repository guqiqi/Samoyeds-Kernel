//
// Created by cpwu on 1/3/24.
//

#ifndef FLEXIBLE_SPMM_MATRIX_UTILS_H
#define FLEXIBLE_SPMM_MATRIX_UTILS_H

#include <iostream>
#include <vector>
#include <algorithm>
#include "utils.h"
#include "assert.h"

template<typename T>
struct DataMatrix {
    std::vector<T> host_ptr;
    T *device_ptr = nullptr;
    std::vector<T> transpose_host_ptr;
    T *transpose_device_ptr = nullptr;
    bool transpose = false;

    int rows = 0, cols = 0;

    virtual void create_data() {}

    void sync_host() {
        if (get_matrix_size() <= 0) return;
        CUDA_CHECK(cudaMemcpy(host_ptr.data(), device_ptr, get_matrix_size() * sizeof(T), cudaMemcpyDeviceToHost));

        if(transpose){
            CUDA_CHECK(cudaMemcpy(transpose_host_ptr.data(), transpose_device_ptr, get_matrix_size() * sizeof(T), cudaMemcpyDeviceToHost));
        }
    }

    void sync_device() {
        if (get_matrix_size() <= 0) return;
        CUDA_CHECK(cudaMemcpy(device_ptr, host_ptr.data(), get_matrix_size() * sizeof(T), cudaMemcpyHostToDevice));
        if(transpose){
            CUDA_CHECK(cudaMemcpy(transpose_device_ptr, transpose_host_ptr.data(), get_matrix_size() * sizeof(T), cudaMemcpyHostToDevice));
        }
    }

    void init_DataMatrix(int rows, int cols, bool transpose=false) {
        this->rows = rows;
        this->cols = cols;
        this->transpose = transpose;
        host_ptr.resize(this->rows * this->cols, 0);
        if (this->rows * this->cols > 0) {
            create_data();
            if(transpose){
                transpose_host_ptr.resize(this->cols * this->rows, 0);
                create_transpose_data();
            }
            CUDA_CHECK(cudaMalloc(&device_ptr, get_matrix_size() * sizeof(T)));
            CUDA_CHECK(cudaMalloc(&transpose_device_ptr, get_matrix_size() * sizeof(T)));
            sync_device();
        }
    }

    uint get_matrix_size() {
        return rows * cols;
    }

    void create_transpose_data(){
        for (int iter_row = 0; iter_row < this->rows; ++iter_row) {
            for (int iter_col = 0; iter_col < this->cols; ++iter_col) {
                this->transpose_host_ptr[iter_col * this->rows + iter_row] = this->host_ptr[iter_row * this->cols + iter_col];
            }
        }
    }

    virtual ~DataMatrix() {
        if (device_ptr) {
            CUDA_CHECK(cudaFree(device_ptr));
            device_ptr = nullptr;
        }
        if (transpose_device_ptr) {
            CUDA_CHECK(cudaFree(transpose_device_ptr));
            transpose_device_ptr = nullptr;
        }
    }
};

template<typename T>
struct ZerosMatrix : public DataMatrix<T> {
public:
    void create_data() {
        for (int i = 0; i < this->get_matrix_size(); i++) {
            this->host_ptr[i] = static_cast<T>(0.0f);
        }
    }
};

template<typename T>
struct RandomMatrix : public DataMatrix<T> {
public:
    void create_data() {
        for (int iter_row = 0; iter_row < this->rows; ++iter_row) {
            for (int iter_col = 0; iter_col < this->cols; ++iter_col) {
                this->host_ptr[iter_row * this->cols + iter_col] = static_cast<T>((float) (std::rand() % 9 - 4));
            }
        }
    }
};

template<typename T>
struct OnesMatrix : public DataMatrix<T> {
public:
    void create_data() {
        for (int i = 0; i < this->get_matrix_size(); i++) {
            this->host_ptr[i] = static_cast<T>(1.0f);
        }
    }
};

template<typename T>
struct OrderMatrix : public DataMatrix<T> {
public:
    void create_data() {
        for (int iter_row = 0; iter_row < this->rows; ++iter_row) {
            for (int iter_col = 0; iter_col < this->cols; ++iter_col) {
                // if (iter_row % 8 >= 2 || iter_row >= 64 || iter_col >= 256)
                if (iter_row % 8 >= 2)
                    this->host_ptr[iter_row * this->cols + iter_col] = static_cast<T>(0);
                else {
                    // this->host_ptr[iter_row * this->cols + iter_col] = static_cast<T>(
                    //         (block_row_idx / 8 * 2 + block_row_idx % 8) * 8 + iter_col + block_num * 128);
                    this->host_ptr[iter_row * this->cols + iter_col] = static_cast<T>(1);
                }
            }
        }
    }
};

template<typename T>
struct SparseMatrix : public DataMatrix<T> {
public:
    // 矩阵是否为行稀疏矩阵
    bool row_sparse = true;

    int indices_len = 0;
    // 压缩矩阵B的索引
    uint *indices_host_ptr = nullptr;
    uint *indices_device_ptr = nullptr;
    // 压缩矩阵B的稠密矩阵
    int condense_rows = 0, condense_cols = 0;
    std::vector<T> condense_host_ptr;
    T *condense_device_ptr = nullptr;

    void create_data() {
        // Create B matrix data
        for (int iter_row = 0; iter_row < this->rows; ++iter_row) {
            for (int iter_col = 0; iter_col < this->cols; ++iter_col) {
                if (iter_col < 64 && iter_row < 32) {
                    // this->host_ptr[iter_row * this->cols + iter_col] = static_cast<T>(iter_row * 8 + iter_col);
                    this->host_ptr[iter_row * this->cols + iter_col] = static_cast<T>(iter_row % 16 + iter_col % 16 +
                                                                                      iter_col / 16);
                } else {
                    this->host_ptr[iter_row * this->cols + iter_col] = static_cast<T>(0);
                }
            }
        }
    }

    void create_data_randomly() {
        // Create B matrix data
        for (int iter_row = 0; iter_row < this->rows; ++iter_row) {
            for (int iter_col = 0; iter_col < this->cols; ++iter_col) {
                this->host_ptr[iter_row * this->cols + iter_col] = static_cast<T>((float) (std::rand() % 9 - 4));
            }
        }
    }

    void create_indices() {
        // Create B indices data
        indices_host_ptr = new uint[indices_len];
        for (int i = 0; i < indices_len; ++i) {
            indices_host_ptr[i] = i * 4;
        }
    }

    //  generate indices randomly (choose a indices out of b)
    void _create_indices_randomly(int a, int b) {
        // Create B indices data
        std::vector<uint> temp;
        temp.reserve(b);
        for (int i = 0; i < b; ++i) {
            temp.push_back(i);
        }
        random_shuffle(temp.begin(), temp.end());
        std::sort(temp.begin(), temp.begin() + a);
        indices_host_ptr = new uint[a];
        for (int i = 0; i < a; ++i) {
            indices_host_ptr[i] = temp[i];
        }
    }

    void create_indices_randomly() {
        if (row_sparse) {
            this->_create_indices_randomly(indices_len, this->rows);
        } else {
            this->_create_indices_randomly(indices_len, this->cols);
        }
    }

    void create_condense_data() {
        // Create dense matrix data
        for (int iter_row = 0; iter_row < this->condense_rows; ++iter_row) {
            for (int iter_col = 0; iter_col < this->condense_cols; ++iter_col) {
                if (row_sparse) {
                    this->condense_host_ptr[iter_row * this->condense_cols + iter_col] = this->host_ptr[
                            this->indices_host_ptr[iter_row] * this->cols + iter_col];
                } else {
                    this->condense_host_ptr[iter_row * this->condense_cols + iter_col] = this->host_ptr[
                            iter_row * this->cols + this->indices_host_ptr[iter_col]];
                }
            }
        }
    }

    void _init_DataMatrix(bool transpose_for_GEMM) {
        this->host_ptr.resize(this->rows * this->cols, 0);
        this->condense_host_ptr.resize(this->condense_rows * this->condense_cols, 0);
        if (row_sparse) {
            this->indices_len = this->condense_rows;
        } else {
            this->indices_len = this->condense_cols;
        }
        // transfer sparse data to device
        // this->create_data();
        // this->create_indices();
        this->create_data_randomly();
        this->create_indices_randomly();
        this->create_condense_data();
        CUDA_CHECK(cudaMalloc(&this->device_ptr, this->get_matrix_size() * sizeof(T)));
        this->sync_device();

        // transfer indices to device
        CUDA_CHECK(cudaMalloc(&this->indices_device_ptr, this->indices_len * sizeof(uint)));
        CUDA_CHECK(cudaMemcpy(this->indices_device_ptr, this->indices_host_ptr, this->indices_len * sizeof(uint),
                              cudaMemcpyHostToDevice));

        std::vector<T> temp;
        if (transpose_for_GEMM) {
            temp.resize(this->condense_rows * this->condense_cols, 0);
            for (int i = 0; i < this->condense_rows; ++i) {
                for (int j = 0; j < this->condense_cols; ++j) {
                    temp[j * this->condense_rows + i] = this->condense_host_ptr[i * this->condense_cols + j];
                }
            }
        } else {
            temp = this->condense_host_ptr;
        }

        // transfer condense data to device
        CUDA_CHECK(cudaMalloc(&this->condense_device_ptr, this->condense_rows * this->condense_cols * sizeof(T)));
        CUDA_CHECK(cudaMemcpy(this->condense_device_ptr, temp.data(),
                              this->condense_rows * this->condense_cols * sizeof(T), cudaMemcpyHostToDevice));
    }
};

// 按行稀疏矩阵
// row = B_indices_len, col = cols
template<typename T>
struct HorizontalSparseMatrix : public SparseMatrix<T> {
public:
    void init_DataMatrix(int rows, int cols, int condense_rows, bool transpose_for_GEMM = false) {
        assert(rows > 0 && cols > 0 && condense_rows > 0);
        assert(condense_rows <= rows);
        this->row_sparse = true;
        this->rows = rows;
        this->cols = cols;
        this->condense_rows = condense_rows;
        this->condense_cols = cols;
        this->indices_len = condense_rows;
        this->_init_DataMatrix(transpose_for_GEMM);
    }
};

// 按列稀疏矩阵
// row = rows, col = B_indices_len
template<typename T>
struct VerticalSparseMatrix : public SparseMatrix<T> {
public:
    void init_DataMatrix(int rows, int cols, int condense_cols, bool transpose_for_GEMM = false) {
        assert(rows > 0 && cols > 0 && condense_cols > 0);
        assert(condense_cols <= cols);
        this->row_sparse = false;
        this->rows = rows;
        this->cols = cols;
        this->condense_rows = rows;
        this->condense_cols = condense_cols;
        this->indices_len = condense_cols;
        this->_init_DataMatrix(transpose_for_GEMM);
    }
};

#endif //FLEXIBLE_SPMM_MATRIX_UTILS_H
