cmake_minimum_required(VERSION 3.16.0)

add_executable(gemv_bench gemv_bench.cc)
target_link_libraries(gemv_bench PUBLIC device_mem_utils data_generator data_verification gemv_rocblas gemv_hipblaslt)

add_executable(allreduce_bench allreduce_bench.cc)
target_link_libraries(allreduce_bench PUBLIC -lrccl)

