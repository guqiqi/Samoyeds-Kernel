set(FORMAT_SOURCES
        format.cpp
        format.hpp
        format_venom.cpp
        format_venom.hpp
        format_ss.cpp
        format_ss.hpp
        formats.hpp)
set_source_files_properties(${FORMAT_SOURCES} PROPERTIES LANGUAGE CUDA)

add_library(format STATIC
        ${FORMAT_SOURCES})