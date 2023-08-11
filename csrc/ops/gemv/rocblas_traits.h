#pragma once

#include <iostream>

#include <hip/hip_runtime.h>
#include <rocblas.h>

#include "gemv_traits.h"

#define ROCBLAS_CHECK(status)                                   \
  {                                                            \
    rocblas_status error = status;                             \
    if (error != rocblas_status_success) {                      \
      std::cerr << "rocBLAS Error: " << error                   \
                << " at: " << __FILE__                         \
                << " " << __LINE__                             \
                << std::endl;                                  \
      exit(EXIT_FAILURE);                                      \
    }                                                          \
  }

template <typename T>
rocblas_status inline rocblasGemmEx(rocblas_handle handle, TGemm<T>& gemm)
{
   rocblas_operation opA = gemm.transA ? rocblas_operation_transpose : rocblas_operation_none;
   rocblas_operation opB = gemm.transB ? rocblas_operation_transpose : rocblas_operation_none;

   ROCBLAS_CHECK(rocblas_gemm_ex(handle,
                             opB, opA,
                             gemm.n, gemm.m, gemm.k,
                             &gemm.alpha,
                             gemm.B, TGemm<T>::Types::hipTypeI, gemm.n,
                             gemm.A, TGemm<T>::Types::hipTypeI, gemm.k,
                             &gemm.beta,
                             gemm.C, TGemm<T>::Types::hipTypeO, gemm.n,
                             gemm.C, TGemm<T>::Types::hipTypeO, gemm.n,
                             rocblas_datatype_f32_r,
                             rocblas_gemm_algo_standard, 0, 0));
   return rocblas_status_success;
}

template <typename T>
void call_rocBLAS(TGemm<T>& gemm, T* h_C_rocblas);

