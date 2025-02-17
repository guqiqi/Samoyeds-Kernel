set(UTIL_SOURCES
        argparse.hpp
        matrix_utils.h
        utils.h
        print_helper.hpp
)
set_source_files_properties(${UTIL_SOURCES} PROPERTIES LANGUAGE CUDA)

add_library(util STATIC
        ${UTIL_SOURCES})