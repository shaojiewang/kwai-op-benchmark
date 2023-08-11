#pragma once

#include <iostream>
#include <string>
#include <stdio.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocblas.h>

#ifdef USE_HIPRAND
#include <hiprand.h>
#endif

#ifdef USE_HIPRAND
#define HIPRAND_CHECK(status)                                    \
  {                                                             \
    hiprandStatus_t error = status;                              \
    if (error != HIPRAND_STATUS_SUCCESS) {                       \
      std::cerr << "HIPRand error: " << error                    \
                << " at: " << __FILE__                          \
                << " " << __LINE__                              \
                << std::endl;                                   \
      exit(EXIT_FAILURE);                                       \
    }                                                           \
  }
#endif

#ifdef USE_HIPRAND
void generate_data(float* d_A, float* d_B, float* d_Bias,
                   half* d_A_fp16, half* d_B_fp16, half* d_Bias_fp16,
                   int size_A, int size_B, int size_Bias);
#endif

template<typename T>
void generate_data(T* d_A, T* d_B, float* d_Bias, float* d_Gain, int size_A, int size_B, int size_Bias, int size_Gain);

