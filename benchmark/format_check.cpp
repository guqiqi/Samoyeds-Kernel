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

#include <iostream>

#include "./format/formats.hpp"
#include "./util/print_helper.hpp"

int main(int argc, const char **argv) {
    std::srand(1);
    // auto *format = new Format_ss<half>(1024, 1024, 1, 16, 2, 8);
    auto *format = new Format_ss<half>(16, 32, 2, 8, 2, 4);
    format->init();

    write_to_file_dec(format->indices, format->get_num_of_rows_indices(), format->get_num_of_cols_indices(),
                      "A_indices");
    write_to_file_hex(format->metadata, format->get_num_of_rows_metadata(), format->get_num_of_cols_metadata(),
                      "A_metadata");
    write_to_file_dec(format->value, format->get_num_of_rows_value(), format->get_num_of_cols_value(),
                      "A_value", 128, 128);
    write_to_file_dec(format->data, format->rows, format->cols,
                      "A_origin", 128, 128);
    write_to_file_dec(format->pruned_value, format->get_num_of_rows_pruned_value(), format->get_num_of_cols_pruned_value(),
                      "A_gemm", 128, 128);

    return 0;
}