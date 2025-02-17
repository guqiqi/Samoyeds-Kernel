//
// Created by cpwu on 2/29/24.
//

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