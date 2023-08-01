
#pragma once
#include <hip/hip_runtime.h>

template<typename T>
void deviceMalloc(T** ptr, size_t size);

template<typename T>
void deviceFree(T*& ptr);


