#pragma once

#include <Tensor.h>
#include <vector>
#include <cassert>

namespace nerd {

template <typename Type>
Tensor<Type> matmult(const Tensor<Type>& A, const Tensor<Type>& B) {
  assert(A.ndim() == 2 && B.ndim() == 2);
  assert(A.get_shape(1) == B.get_shape(0));
  std::size_t inner_dim = B.get_shape(0);
  std::size_t A_rows = A.get_shape(0);
  std::size_t B_cols = B.get_shape(1);

  Tensor<Type> res(Shape{A_rows, B_cols}, Type{0});

  #pragma omp parallel
  {
    #pragma omp for
    for (std::size_t i = 0; i < A_rows; ++i) {
      std::vector<Type> buffer(B_cols);
      for (std::size_t k = 0; k < inner_dim; ++k) {
        Type a_ik = A(i, k);  
        #pragma omp simd
        for (std::size_t j = 0; j < B_cols; ++j) {
          buffer[j] += a_ik * B(k, j);
        }
      }

      for (std::size_t j = 0; j < B_cols; ++j) {
        res(i, j) = buffer[j];
      }
    }
  }
  return res;
}

}
