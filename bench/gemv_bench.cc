#include <stdio.h>
#include <iostream>

#include "csrc/utils/device_mem_utils.h"
#include "csrc/utils/data_generator.h"
#include "csrc/utils/data_verification.h"

#include "csrc/ops/gemv/rocblas_traits.h"
#include "csrc/ops/gemv/hipblaslt_traits.h"

int main(int argc, char* argv[]){

    if(argc <= 3){
        std::cout << "at least give this bench 3 args, like 1024 2048 512, which means M, N, K are 1024 2048 512" << std::endl;
        return 0;
    }

    int32_t M, N, K;
    M = atoi(argv[1]);
    N = atoi(argv[2]);
    K = atoi(argv[3]);

    bool transA = (argc >= 5 ? (atoi(argv[4]) != 0) : false);
    bool transB = (argc >= 6 ? (atoi(argv[5]) != 0) : false);

    printf("\nM = %d, N = %d, K = %d \n\n", M, N, K);

    float *d_Bias, *d_Gain;
    int64_t size_A = M * K, size_B = K * N, size_C = M * N, size_Bias = N, size_Gain = N;

    half *d_A_fp16, *d_B_fp16, *d_C_fp16, *d_Bias_fp16, *d_Gain_fp16;
    deviceMalloc(&d_A_fp16, size_A);
    deviceMalloc(&d_B_fp16, size_B);
    deviceMalloc(&d_C_fp16, size_C);
    deviceMalloc(&d_Bias, size_Bias);
    deviceMalloc(&d_Gain, size_Gain);

    generate_data(d_A_fp16, d_B_fp16, d_Bias, d_Gain,
        size_A, size_B, size_Bias, size_Gain);

    half* h_C_rocblas_fp16 = (half*)malloc(size_C * sizeof(half));
    if (h_C_rocblas_fp16 == nullptr) {
      printf("h_C_blaslt_fp16 is nullptr. \n");
    } else {
      memset(h_C_rocblas_fp16, 0, size_C * sizeof(half));
    }

    half* h_C_hipblaslt_fp16 = (half*)malloc(size_C * sizeof(half));
    if (h_C_hipblaslt_fp16 == nullptr) {
      printf("h_C_blaslt_fp16 is nullptr. \n");
    } else {
      memset(h_C_hipblaslt_fp16, 0, size_C * sizeof(half));
    }

    TGemm<half> gemm_fp16(M, N, K, d_A_fp16, d_B_fp16, d_C_fp16, transA, transB);
    call_rocBLAS(gemm_fp16, h_C_rocblas_fp16);
    call_hipblaslt(gemm_fp16, h_C_hipblaslt_fp16);

    verify(h_C_hipblaslt_fp16, h_C_rocblas_fp16, M * N, "hipblaslt");


    deviceFree(d_A_fp16);
    deviceFree(d_B_fp16);
    deviceFree(d_C_fp16);

    free(h_C_rocblas_fp16);
    free(h_C_hipblaslt_fp16);


    std::cout << "gemv test" << std::endl;
    return 0;

}
