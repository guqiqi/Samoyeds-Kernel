set(SSMM_SOURCES
        horizontal_ssmm_kernel.h
        horizontal_ssmm_kernel_op.h
        data_structure_helper.h
        copy_helper.h
        mma_helper.h
)
set_source_files_properties(${SSMM_SOURCES} PROPERTIES LANGUAGE CUDA)

add_library(ssmm STATIC
        ${SSMM_SOURCES})