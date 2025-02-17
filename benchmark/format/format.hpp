//
// Created by CPWu2017@126.com on 12/22/23.
//

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
