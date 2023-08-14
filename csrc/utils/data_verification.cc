#include "data_verification.h"

template<typename T>
void verify(const T* lhs, const T* rhs, int size, const std::string& str) {
  double max_err = std::numeric_limits<double>::min();
  double rtol = 1e-3;
  double atol = 1e-5;
  int errors = 0;

  float max_err_ref, max_err_out;
  std::cout << "Verifying " << str << " ..." << size << std::endl;
  for (int i = 0; i < size; i++) {
    float ref = lhs[i];
    float out = rhs[i];
    double err = std::abs(out - ref);
    if(err > atol + rtol * std::abs(ref) || !std::isfinite(out) || !std::isfinite(ref)) {
      if (err > max_err) {
        max_err = err;
        max_err_ref = ref;
        max_err_out = out;
      }
      errors++;
      // if (errors <= 16) printf("i: %d    %f %f\n", i, ref, out);
    }
  }
  std::cout << "    Errors: "<< errors
            << std::setw(12) << std::setprecision(7)
            << ", max error: " << max_err
            << " ref: " << max_err_ref
            << " out: " << max_err_out << std::endl;
}

template void verify<float>(const float* lhs, const float* rhs, int size, const std::string& str);
template void verify<half>(const half* lhs, const half* rhs, int size, const std::string& str);

