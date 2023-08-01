cmake_minimum_required(VERSION 3.16.0)

project(ops_benchmark LANGUAGES CXX HIP)

set(ROCM_HOME /opt/rocm)
list(APPEND CMAKE_PREFIX_PATH /opt/rocm ${ROCM_HOME}/hip ${ROCM_HOME}/hcc ${ROCM_HOME}/miopen ${ROCM_HOME}/hiprand ${ROCM_HOME}/rocrand)

set(ROCM_SYMLINK_LIBS OFF)
find_package(ROCM REQUIRED PATHS /opt/rocm)

## C++
enable_language(CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
message("CMAKE_CXX_COMPILER_ID: ${CMAKE_CXX_COMPILER_ID}")

## HIP
message(STATUS "Build with HIP ${hip_VERSION}")
find_package(HIP REQUIRED)
find_library(HIP_LIB amdhip64 REQUIRED)
find_library(ROC_BLAS rocblas REQUIRED)
SET(USE_HIPRAND OFF CACHE BOOL "USE_HIPRAND")
if(USE_HIPRAND)
    find_library(HIPRAND_LIB hiprand REQUIRED)
endif()

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++17 -lhipblas -L /opt/rocm/lib -L /usr/local/lib -O3 -fPIE --offload-arch=gfx90a")

SET(BUILD_DEV OFF CACHE BOOL "BUILD_DEV")
if(BUILD_DEV)
    string(APPEND CMAKE_CXX_FLAGS "--save-temps -Werror -Weverything")
endif()
message("CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")

set(ROCM_INCLUDES
if(USE_HIPRAND)
    /opt/rocm/hiprand/include
    /opt/rocm/rocrand/include
endif()
)

add_subdirectory(bench)

