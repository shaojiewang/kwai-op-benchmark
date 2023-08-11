#pragma once

#include <iostream>
#include <string>
#include <stdio.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocblas.h>

constexpr int NUM_ITERATIONS = 10;
constexpr int NUM_THREADS_PER_BLOCK = 256;

template <class INT, class INT2>
inline __host__ __device__ INT CeilDiv(INT a, INT2 b)  // ceil(a/b)
{
  return (INT)(((size_t)a + (size_t)b - 1) / (size_t)b);  // these size_t casts are necessary since b may be INT_MAX (for maxGridSize[])
}

template <typename T>
struct TGemmTypes
{
};

template <>
struct TGemmTypes<half>
{
    static const rocblas_datatype hipTypeI = rocblas_datatype_f16_r;
    using dataTypeI = half;
    static const rocblas_datatype hipTypeO = rocblas_datatype_f16_r;
    using dataTypeO = half;
    static const rocblas_datatype hipTypeS = rocblas_datatype_f32_r; // scale type
    using dataTypeS = float;
};

template <>
struct TGemmTypes<float>
{
    static const rocblas_datatype hipTypeI = rocblas_datatype_f32_r;
    using dataTypeI = float;
    static const rocblas_datatype hipTypeO = rocblas_datatype_f32_r;
    using dataTypeO = float;
    static const rocblas_datatype hipTypeS = rocblas_datatype_f32_r; // scale type
    using dataTypeS = float;
};


template <typename T>
struct TGemm
{
    int m, n, k, ldA, ldB, ldC, rA, rB, rC, cA, cB, cC;
    size_t elemA;
    size_t elemB;
    size_t elemC;
  
    size_t bytesA;
    size_t bytesB;
    size_t bytesC;

    using Types = TGemmTypes<T>;
    typename Types::dataTypeI* A{nullptr};
    typename Types::dataTypeI* B{nullptr};
    typename Types::dataTypeO* C{nullptr};

    bool transA, transB;

    typename Types::dataTypeS alpha;
    typename Types::dataTypeS beta;

    TGemm() {}

    // Row Major
    TGemm(int m_, 
          int n_, 
          int k_,
          typename Types::dataTypeI* A_,
          typename Types::dataTypeI* B_,
          typename Types::dataTypeO* C_,
          bool transA_ = false, 
          bool transB_ = false)
    {
        m = m_;
        n = n_;
        k = k_;
        elemA = m * k;
        elemB = n * k;
        elemC = m * n;
        bytesA = sizeof(T) * elemA;
        bytesB = sizeof(T) * elemB;
        bytesC = sizeof(T) * elemC;
  
        A = A_;
        B = B_;
        C = C_;
  
        transA = transA_;
        transB = transB_;
        ldA = transA ? m : k;
        ldB = transB ? k : n;
        ldC = n;

        alpha = T(1.f);
        beta = T(0.f);
    }
};

