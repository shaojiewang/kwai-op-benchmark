#include <hipblaslt/hipblaslt.h>

#include <hip/hip_runtime.h>

#define HIPBLASLT_CHECK(status)                          \
    {                                                    \
        hipblasStatus_t error = status;                  \
        if(error != HIPBLAS_STATUS_SUCCESS){             \
            std::cerr << "hipblaslt Error: " << error    \
                      << " at: " << __FILE__             \
                      << " " << __LINE__                 \
                      << std::endl;                      \
            exit(EXIT_FAILURE);                          \
        }                                                \
    }

// For large K and small M/N, K dim will be split to multiple workgroups and buffers,
// which will require additional workspace. Here we set the max workspace size to 32MB.
constexpr const size_t kHipBlasLtMaxWorkSpaceSizeInBytes = 32 * 1024 * 1024;
// We only keep one heuristic result here. Note that for tuned input sizes, the first result
// will be the most performant one; but in untuned cases, this is not guaranteed.
constexpr const int kHeuristicResultCount = 1;

enum ActivationType {
  NONE = 0,
  RELU = 1,
  GELU = 2,
};

template <typename T>
constexpr hipblasDatatype_t HipBlasDataTypeFor(const T*);

template <>
constexpr hipblasDatatype_t HipBlasDataTypeFor(const float*) {
  return HIPBLAS_R_32F;
}

template <>
constexpr hipblasDatatype_t HipBlasDataTypeFor(const half*) {
  return HIPBLAS_R_16F;
}

#if 0
template <>
constexpr hipblasDatatype_t HipBlasDataTypeFor(const BFloat16*) {
  return HIPBLAS_R_16B;
}
#endif

template <>
constexpr hipblasDatatype_t HipBlasDataTypeFor(const double*) {
  return HIPBLAS_R_64F;
}

#if 0
template <typename T>
hipblasStatus_t inline hipblasltGemm(hipblasLtHandle_t handle, TGemm<T>& gemm)
{
    hipblasOperation_t trans_a = gemm.transA ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    hipblasOperation_t trans_b = gemm.transB ? HIPBLAS_OP_T : HIPBLAS_OP_N;

    HIPBLASLT_CHECK();
    
}
#endif

template <typename T>
void call_hipblaslt(TGemm<T>& gemm, T* h_C_hipblaslt);

