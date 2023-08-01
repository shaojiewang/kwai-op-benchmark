cmake_minimum_required(VERSION 3.16.0)

add_executable(gemv_bench gemv_bench.cc)
target_link_libraries(gemv_bench PUBLIC device_mem_utils)

