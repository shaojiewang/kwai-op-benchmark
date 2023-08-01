#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "csrc/utils/device_mem_utils.h"
#include "csrc/utils/device_utils.h"

template<typename T>
void deviceMalloc(T** ptr, size_t size){
#if defined(__HIPCC__)
    device_check_error(hipMalloc(ptr, sizeof(T) * size));
#endif
}

template<typename T>
void deviceFree(T*& ptr){
    if (ptr != nullptr) {
#if defined(__HIPCC__)
        device_check_error(hipFree(ptr));
#endif
        ptr == nullptr;
    }
}

