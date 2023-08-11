#include <cstdint>
#include <string>
#include <stdio.h>

#include "hip/hip_runtime.h"

#include "csrc/utils/device_utils.h"

#include "gemv_traits.h"
#include "rocblas_traits.h"

constexpr int NUM_ELEMENTS_PER_THREAD = 4;

struct fast_divmod {
  fast_divmod(int d = 1) {
    d_ = d == 0 ? 1 : d;
    // ORT_ENFORCE(d_ >= 1 && d_ <= static_cast<uint32_t>(std::numeric_limits<int>::max()));

    for (l_ = 0; l_ < 32; l_++) if ((1U << l_) >= d_) break;

    uint64_t one = 1;
    uint64_t m = ((one << 32) * ((one << l_) - d_)) / d_ + 1;
    M_ = static_cast<uint32_t>(m);
    // according to paper, the value of m' should fit in a unsigned integer.
    // ORT_ENFORCE(M_ > 0 && M_ == m);
  }

  __host__ __device__ inline int div(int n) const {
#ifdef __CUDA_ARCH__
    uint32_t t = __umulhi(M_, n);
    return (t + n) >> l_;
#else
    // Using uint64_t for t, then t + n won't overflow.
    uint64_t t = ((uint64_t) M_ * n) >> 32;
    return static_cast<int>((t + n) >> l_);
#endif
  }

  __host__ __device__ inline int mod(int n) const {
    return n - div(n) * d_;
  }

  __host__ __device__ inline void divmod(int n, int& q, int& r) const {
    q = div(n);
    r = n - q * d_;
  }

  uint32_t d_;  // divisor
  uint32_t M_;  // m' in the paper.
  uint32_t l_;  // l_ = ceil(log2(d_))
};

template <typename T, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void AddBias(const T* lhs, const T* rhs,
                        const fast_divmod fdm_H,
                        const fast_divmod fdm_C,
                        T* output,
                        int32_t N) {
  int32_t start = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;
  T lvalue[NumElementsPerThread];
  T rvalue[NumElementsPerThread];

  int32_t id = start;
#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      int32_t rhs_id = fdm_H.div(id);
      int q, r;
      fdm_C.divmod(rhs_id, q, r);
      rhs_id = r;

      lvalue[i] = lhs[id];
      rvalue[i] = rhs[rhs_id];

      id += NumThreadsPerBlock;
    }
  }

  id = start;
#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      output[id] = lvalue[i] + rvalue[i];

      id += NumThreadsPerBlock;
    }
  }
}

template <typename T>
void call_rocBLAS(TGemm<T>& gemm, T* h_C_rocblas) {
  std::cout << "\nRunning with rocBLAS " << (std::is_same<T, half>::value ? "FP16..." : "FP32...") << std::endl;
  device_check_error(hipMemset(gemm.C, 0, gemm.elemC * sizeof(T)));

  // float* d_zero;
  // int zero_size = 256 * 1024 * 1024;
  // device_check_error(hipMalloc((void**)&d_zero,  zero_size));

  rocblas_handle handle;
  ROCBLAS_CHECK(rocblas_create_handle(&handle));
  ROCBLAS_CHECK(rocblasGemmEx(handle, gemm));

  hipEvent_t start, stop;
  device_check_error(hipEventCreate(&start));
  device_check_error(hipEventCreate(&stop));

  float total_time = 0.0f;
  for (int i = 0; i < NUM_ITERATIONS; ++i) {
    // device_check_error(hipMemset(d_zero, 0, zero_size));
    device_check_error(hipEventRecord(start));
    ROCBLAS_CHECK(rocblasGemmEx(handle, gemm));
    device_check_error(hipEventRecord(stop));
    device_check_error(hipEventSynchronize(stop));

    float time = 0.0f;
    device_check_error(hipEventElapsedTime(&time, start, stop));
    total_time += time;
  }

  printf("rocBLAS time:  %3.4f ms \n", total_time / NUM_ITERATIONS);

  device_check_error(hipMemcpy(h_C_rocblas, gemm.C, gemm.elemC * sizeof(T), hipMemcpyDeviceToHost));
  device_check_error(hipDeviceSynchronize());

  device_check_error(hipEventDestroy(start));
  device_check_error(hipEventDestroy(stop));
  ROCBLAS_CHECK(rocblas_destroy_handle(handle));
}

#define INSTANTIATE_CUBLAS(T) template void call_rocBLAS<T>(TGemm<T>& gemm, T* h_C_rocblas);
INSTANTIATE_CUBLAS(float)
INSTANTIATE_CUBLAS(half)

