#pragma once

#include <iostream>
#include <stdio.h>
#include <algorithm>
#include <string>
#include <iomanip>

#include "device_utils.h"

template <typename T>
void verify(const T* lhs, const T* rhs, size_t size, const std::string& str);

