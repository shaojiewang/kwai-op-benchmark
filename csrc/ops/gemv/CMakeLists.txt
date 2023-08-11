add_library(gemv_rocblas STATIC gemv_rocblas.cc)
target_link_libraries(gemv_rocblas rocblas)

add_library(gemv_hipblaslt STATIC gemv_hipblaslt.cc)
target_link_libraries(gemv_hipblaslt -lhipblaslt device_mem_utils)
