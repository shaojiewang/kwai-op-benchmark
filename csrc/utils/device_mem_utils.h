
#pragma once
#include "csrc/utils/device_utils.h"


template<typename T>
void deviceMalloc(T** ptr, size_t size);

template<typename T>
void deviceFree(T*& ptr);


