#include <hip/hip_runtime.h>

#if defined(__HIPCC__)
static const char* _deviceGetErrorEnum(hipError_t error)
{
    return hipGetErrorString(error);
}
#endif

template<typename T>
void check(T result, char const* const func, const char* const file, int const line)
{
    if (result) {
        throw std::runtime_error(std::string("[FT][ERROR] device runtime error: ") + (_deviceGetErrorEnum(result)) + " "
                                 + file + ":" + std::to_string(line) + " \n");
    }
}

#define check_device_error(val) check((val), #val, __FILE__, __LINE__)