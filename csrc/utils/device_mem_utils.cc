#include "csrc/utils/device_mem_utils.h"

template<typename T>
void deviceMalloc(T** ptr, size_t size){
#if defined(__HIPCC__)
    device_check_error(hipMalloc(ptr, sizeof(T) * size));
#endif
}

template void deviceMalloc(float** ptr, size_t size);
template void deviceMalloc(half** ptr, size_t size);
template void deviceMalloc(uint16_t** ptr, size_t size);
template void deviceMalloc(int** ptr, size_t size);
template void deviceMalloc(bool** ptr, size_t size);
template void deviceMalloc(char** ptr, size_t size);
template void deviceMalloc(int8_t** ptr, size_t size);

template<typename T>
void deviceFree(T*& ptr){
    if (ptr != nullptr) {
#if defined(__HIPCC__)
        device_check_error(hipFree(ptr));
#endif
        ptr == nullptr;
    }
}

template void deviceFree(float*& ptr);
template void deviceFree(half*& ptr);

