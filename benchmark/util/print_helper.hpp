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

#ifndef FLEXIBLE_SPMM_PRINT_HELPER_HPP
#define FLEXIBLE_SPMM_PRINT_HELPER_HPP

#include <iomanip>
#include <iostream>
#include <cuda_fp16.h>
#include <vector>
#include <fstream>
#include <string>
#include <filesystem>

void output_format_half_hex(half a) {
    std::cout << std::hex << std::setw(4) << std::setfill('0') << reinterpret_cast<unsigned short &>(a) << std::dec;
}

void output_format_half_dec(half a) {
    std::cout << static_cast<float>(a);
}

void write_to_file_hex(std::vector<uint> &meta, int row, int col, std::string filename, int write_row = 16, int write_col = 8) {
    write_row = write_row > row ? row : write_row;
    write_col = write_col > col ? col : write_col;
    auto dirName = std::filesystem::current_path().string() + "/csv_dir/";
    if (!std::filesystem::exists(dirName)) {
        std::filesystem::create_directories(dirName);
    }
    std::ofstream outputFile(dirName + filename + ".csv");  // 打开或创建一个名为 "filename.csv" 的文件

    if (outputFile.is_open()) {  // 检查文件是否成功打开
        for (int i = 0; i < row && i < write_row; i++) {
            for (int j = 0; j < col && j < write_col; j++) {
                // half floatValue = reinterpret_cast<half&>(a[i*col+j]);
                // unsigned short hexValue = reinterpret_cast<unsigned short &>(meta[i * col + j]);
                uint hexValue = meta[i * col + j];
                outputFile << "0x" << std::hex << std::setw(8) << std::setfill('0') << hexValue << "," << std::dec;
            }
            outputFile << std::endl;
        }
        outputFile.close();  // 关闭文件
        std::cout << "File " << filename << " written successfully." << std::endl;
    } else {
        std::cout << "Failed to open the file " << filename << "." << std::endl;
    }
}

void
write_to_file_dec(std::vector<half> &a, int row, int col, std::string filename, int write_row = 16, int write_col = 8) {
    write_row = write_row > row ? row : write_row;
    write_col = write_col > col ? col : write_col;
    auto dirName = std::filesystem::current_path().string() + "/csv_dir/";
    if (!std::filesystem::exists(dirName)) {
        std::filesystem::create_directories(dirName);
    }
    std::ofstream outputFile(dirName + filename + ".csv");

    if (outputFile.is_open()) {  // 检查文件是否成功打开
        for (int i = 0; i < row && i < write_row; i++) {
            for (int j = 0; j < col && j < write_col; j++) {
                // half floatValue = reinterpret_cast<half&>(a[i*col+j]);
                float floatValue = static_cast<float>(a[i * col + j]);
                if (j != col - 1 && j != write_col - 1) {
                    outputFile << floatValue << ",";
                } else {
                    outputFile << floatValue;
                }
            }
            outputFile << std::endl;
        }
        outputFile.close();  // 关闭文件
        std::cout << "File " << filename << " written successfully." << std::endl;
    } else {
        std::cout << "Failed to open the file " << filename << "." << std::endl;
    }
}

void
write_to_file_dec(std::vector<uint> &a, int row, int col, std::string filename, int write_row = 16, int write_col = 8) {
    write_row = write_row > row ? row : write_row;
    write_col = write_col > col ? col : write_col;
    auto dirName = std::filesystem::current_path().string() + "/csv_dir/";
    if (!std::filesystem::exists(dirName)) {
        std::filesystem::create_directories(dirName);
    }
    std::ofstream outputFile(dirName + filename + ".csv");

    if (outputFile.is_open()) {  // 检查文件是否成功打开
        for (int i = 0; i < row && i < write_row; i++) {
            for (int j = 0; j < col && j < write_col; j++) {
                // half floatValue = reinterpret_cast<half&>(a[i*col+j]);
                float floatValue = static_cast<float>(a[i * col + j]);
                if (j != col - 1 && j != write_col - 1) {
                    outputFile << floatValue << ",";
                } else {
                    outputFile << floatValue;
                }
            }
            outputFile << std::endl;
        }
        outputFile.close();  // 关闭文件
        std::cout << "File " << filename << " written successfully." << std::endl;
    } else {
        std::cout << "Failed to open the file " << filename << "." << std::endl;
    }
}


#endif //FLEXIBLE_SPMM_PRINT_HELPER_HPP
