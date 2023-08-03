#include "hip/hip_runtime.h"
#include <cstdint>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <iomanip>

#include "data_generator.h"
#include "device_utils.h"

template <typename T, typename S>
__global__ void cast(T *out, S *in, int n) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < n) {
     out[idx] = in[idx];
  }
}

#ifdef USE_HIPRAND
void generate_data(float* d_A, float* d_B, float* d_Bias,
                   half* d_A_fp16, half* d_B_fp16, half* d_Bias_fp16,
                   int size_A, int size_B, int size_Bias) {
  // Generate random number for A, B
  hiprandGenerator_t gen;
  HIPRAND_CHECK(hiprandCreateGenerator(&gen, HIPRAND_RNG_PSEUDO_DEFAULT));
  HIPRAND_CHECK(hiprandSetPseudoRandomGeneratorSeed(gen, 1337ULL));
  HIPRAND_CHECK(hiprandGenerateUniform(gen, d_A, size_A));
  HIPRAND_CHECK(hiprandGenerateUniform(gen, d_B, size_B));
  HIPRAND_CHECK(hiprandGenerateUniform(gen, d_Bias, size_Bias));
  HIPRAND_CHECK(hiprandDestroyGenerator(gen));

  // hiprand doesn't currently support fp16 so we generate in fp32 and convert to fp16.
  int blocksPerGrid = static_cast<int>(CeilDiv(size_A, NUM_THREADS_PER_BLOCK));
  hipLaunchKernelGGL(cast, blocksPerGrid, NUM_THREADS_PER_BLOCK , 0, 0, d_A_fp16, d_A, size_A);

  blocksPerGrid = static_cast<int>(CeilDiv(size_B, NUM_THREADS_PER_BLOCK));
  hipLaunchKernelGGL(cast, blocksPerGrid, NUM_THREADS_PER_BLOCK , 0, 0, d_B_fp16, d_B, size_B);

  blocksPerGrid = static_cast<int>(CeilDiv(size_Bias, NUM_THREADS_PER_BLOCK));
  hipLaunchKernelGGL(cast, blocksPerGrid, NUM_THREADS_PER_BLOCK , 0, 0, d_Bias_fp16, d_Bias, size_Bias);
  // hipLaunchKernelGGL(fill, blocksPerGrid, NUM_THREADS_PER_BLOCK , 0, 0, d_Bias_fp16, 1.0, size_Bias);
  device_check_error(hipDeviceSynchronize());
}
#endif

template <typename T>
struct Generator
{
    float min_value = 0;
    float max_value = 1;

    T next()
    {
        float tmp = float(std::rand()) / float(RAND_MAX);

        return static_cast<T>(min_value + tmp * (max_value - min_value));
    }
};

template<typename T>
void generate_data(T* d_A, T* d_B, float* d_Bias, float* d_Gain,
                   int size_A, int size_B, int size_Bias, int size_Gain) {
  // Generate random number for A, B
  T* h_A = (T*)malloc(size_A * sizeof(T));
  if (h_A == nullptr) {
    printf("h_A is nullptr. \n");
  } else {
    memset(h_A, 0, size_A * sizeof(T));
  }

  T* h_B = (T*)malloc(size_B * sizeof(T));
  if (h_B == nullptr) {
    printf("h_B is nullptr. \n");
  } else {
    memset(h_B, 0, size_B * sizeof(T));
  }

  float* h_Bias = (float*)malloc(size_Bias * sizeof(float));
  if (h_Bias == nullptr) {
    printf("h_Bias is nullptr. \n");
  } else {
    memset(h_Bias, 0, size_Bias * sizeof(float));
  }

  float* h_Gain = (float*)malloc(size_Gain * sizeof(float));
  if (h_Gain == nullptr) {
    printf("h_Gain is nullptr. \n");
  } else {
    memset(h_Gain, 0, size_Gain * sizeof(float));
  }

  Generator<T> gen_A{0.0, 1.0};
  for (int i = 0; i < size_A; ++i) {
    h_A[i] = gen_A.next();
  }

  Generator<T> gen_B{-0.5, 0.5};
  for (int i = 0; i < size_B; ++i) {
    h_B[i] = gen_B.next();
  }

  Generator<float> gen_Bias{-0.5, 0.5};
  for (int i = 0; i < size_Bias; ++i) {
    // h_Bias[i] = gen_Bias.next();
    h_Bias[i] = 0.0;
  }

  Generator<float> gen_Gain{0.0, 1.0};
  for (int i = 0; i < size_Gain; ++i) {
    // h_Gain[i] = gen_Gain.next();
    h_Gain[i] = 1.0;
  }

  device_check_error(hipMemcpy(d_A, h_A, size_A * sizeof(T), hipMemcpyHostToDevice));
  device_check_error(hipMemcpy(d_B, h_B, size_B * sizeof(T), hipMemcpyHostToDevice));
  device_check_error(hipMemcpy(d_Bias, h_Bias, size_Bias * sizeof(float), hipMemcpyHostToDevice));
  device_check_error(hipMemcpy(d_Gain, h_Gain, size_Gain * sizeof(float), hipMemcpyHostToDevice));

  device_check_error(hipDeviceSynchronize());

  free(h_A);
  free(h_B);
  free(h_Bias);
  free(h_Gain);
}

template void generate_data<half>(half* d_A, half* d_B, float* d_Bias, float* d_Gain, int size_A, int size_B, int size_Bias, int size_Gain);

