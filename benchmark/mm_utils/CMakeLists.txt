set(MM_UTIL_SOURCES
        spmm_utils.h
        memcpy_pipeline.h
        memcpy_util.h
)
set_source_files_properties(${MM_UTIL_SOURCES} PROPERTIES LANGUAGE CUDA)

add_library(mm_util STATIC
        ${MM_UTIL_SOURCES})