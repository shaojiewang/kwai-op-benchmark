#include <cstdint>
#include <string>
#include <stdio.h>

#include "hip/hip_runtime.h"

#include "csrc/utils/device_utils.h"
#include "csrc/utils/device_mem_utils.h"

#include "gemv_traits.h"
#include "rocblas_traits.h"
#include "hipblaslt_traits.h"

#define NUM_ITERATIONS 4

template <typename T>
void call_hipblaslt(TGemm<T>& gemm, T* h_C_hipblaslt) {
    std::cout << "\nRunning with hipblaslt " << (std::is_same<T, half>::value ? "FP16..." : "FP32...") << std::endl;
    device_check_error(hipMemset(gemm.C, 0, gemm.elemC * sizeof(T)));

    hipblasLtHandle_t handle;

    HIPBLASLT_CHECK(hipblasLtCreate(&handle));

    hipblasDatatype_t Atype_ = HIPBLAS_R_16F;
    hipblasDatatype_t Btype_ = HIPBLAS_R_16F;
    hipblasDatatype_t Ctype_ = HIPBLAS_R_16F;

    hipblasOperation_t transa = gemm.transA ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    hipblasOperation_t transb = gemm.transB ? HIPBLAS_OP_T : HIPBLAS_OP_N;

    hipblasLtMatmulDesc_t   operationDesc = NULL;
    hipblasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    hipblasDatatype_t       scaleType = HIPBLAS_R_32F;
    hipblasLtComputeType_t  computeType;

    half h_alpha = gemm.alpha;
    half h_beta = gemm.beta;

    const void* alpha = reinterpret_cast<void*>(&h_alpha);
    const void* beta  = reinterpret_cast<void*>(&h_beta);

    computeType = HIPBLASLT_COMPUTE_F32;

    int m = gemm.m;
    int n = gemm.n;
    int k = gemm.k;

    int lda = gemm.ldA;
    int ldb = gemm.ldB;
    int ldc = gemm.ldC;
    
    
    // Create descriptors for the original matrices
    hipblasLtMatrixLayoutCreate(&Adesc, Atype_, transa == HIPBLAS_OP_N ? k : m, transa == HIPBLAS_OP_N ? m : k, lda);
    hipblasLtMatrixLayoutCreate(&Bdesc, Btype_, transb == HIPBLAS_OP_N ? n : k, transb == HIPBLAS_OP_N ? k : n, ldb);
    hipblasLtMatrixLayoutCreate(&Cdesc, Ctype_, n, m, ldc);
 
    hipblasLtMatmulDescCreate(&operationDesc, computeType, scaleType);
    hipblasLtMatmulDescSetAttribute(operationDesc, HIPBLASLT_MATMUL_DESC_TRANSA, &transb, sizeof(hipblasOperation_t));
    hipblasLtMatmulDescSetAttribute(operationDesc, HIPBLASLT_MATMUL_DESC_TRANSB, &transa, sizeof(hipblasOperation_t));

    hipblasLtMatmulAlgo_t algo;
    int                   workspaceSize = kHipBlasLtMaxWorkSpaceSizeInBytes; 
    T*                    workSpace     = nullptr;
    deviceMalloc(&workSpace, workspaceSize);

    hipblasLtMatmulPreference_t pref;
    size_t max_workspace_size = workspaceSize;
    hipblasLtMatmulPreferenceCreate(&pref);
    hipblasLtMatmulPreferenceSetAttribute(
        pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_workspace_size, sizeof(max_workspace_size));

    hipblasLtMatmulHeuristicResult_t heuristic_result[kHeuristicResultCount] = {};
    int ret_algo_count = 0;
    HIPBLASLT_CHECK(hipblasLtMatmulAlgoGetHeuristic(handle,
                                                     operationDesc,
                                                     Bdesc,
                                                     Adesc,
                                                     Cdesc,
                                                     Cdesc,
                                                     pref,
                                                     kHeuristicResultCount,
                                                     heuristic_result,
                                                     &ret_algo_count));

    assert(ret_algo_count > 0); 
    algo = heuristic_result[0].algo;
    hipblasLtMatmulPreferenceDestroy(pref); 

    hipEvent_t start, stop;
    device_check_error(hipEventCreate(&start));
    device_check_error(hipEventCreate(&stop));

    printf("m=%d, n=%d, k=%d, lda=%d, ldb=%d, ldc=%d, wss=%d\n",
        m, n, k, lda, ldb, ldc, workspaceSize);

    // warm up
    for(int i = 0; i < NUM_ITERATIONS; i++){
        HIPBLASLT_CHECK(hipblasLtMatmul(handle,
                       operationDesc,
                       alpha,
                       gemm.B,
                       Bdesc,
                       gemm.A,
                       Adesc,
                       beta,
                       gemm.C,
                       Cdesc,
                       gemm.C,
                       Cdesc,
                       &algo,
                       workSpace,
                       workspaceSize,
                       0));
    }
    
    float total_time = 0.0f;
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        // device_check_error(hipMemset(d_zero, 0, zero_size));
        device_check_error(hipEventRecord(start));

        HIPBLASLT_CHECK(hipblasLtMatmul(handle,
                       operationDesc,
                       alpha,
                       gemm.B,
                       Bdesc,
                       gemm.A,
                       Adesc,
                       beta,
                       gemm.C,
                       Cdesc,
                       gemm.C,
                       Cdesc,
                       &algo,
                       workSpace,
                       workspaceSize,
                       0));

        device_check_error(hipEventRecord(stop));       
        device_check_error(hipEventSynchronize(stop));

        float time = 0.0f;
        device_check_error(hipEventElapsedTime(&time, start, stop));
        total_time += time;
        printf("%dth time=%f\n", i, time);
    }

    hipblasLtMatmulDescDestroy(operationDesc);
    hipblasLtMatrixLayoutDestroy(Adesc);
    hipblasLtMatrixLayoutDestroy(Bdesc);
    hipblasLtMatrixLayoutDestroy(Cdesc);

    printf("hipblasLt time:  %3.4f ms \n", total_time / NUM_ITERATIONS);

    device_check_error(hipMemcpy(h_C_hipblaslt, gemm.C, gemm.elemC * sizeof(T), hipMemcpyDeviceToHost));
    device_check_error(hipDeviceSynchronize());

    device_check_error(hipEventDestroy(start));
    device_check_error(hipEventDestroy(stop));
    deviceFree(workSpace);
}

#define INSTANTIATE_CUBLAS(T) template void call_hipblaslt<T>(TGemm<T>& gemm, T* h_C_rocblas);
INSTANTIATE_CUBLAS(float)
INSTANTIATE_CUBLAS(half)

     


