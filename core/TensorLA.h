#pragma once

#include <Tensor.h>
#include <cassert>

namespace nerd {

template <typename Type>
Tensor<Type> matmult(const Tensor<Type>& A, const Tensor<Type>& B) {
  assert(A.ndim() == 2 && B.ndim() == 2);
  assert(A.get_shape(1) == B.get_shape(0));
  std::size_t A_rows = A.get_shape(0);
  std::size_t B_cols = B.get_shape(1);
  Tensor<Type> res(Shape{A_rows, B_cols});
  for (std::size_t i = 0; i < A_rows; ++i) {
    for (std::size_t j = 0; j < B_cols; ++j) {
      for (std::size_t k = 0; k < B.get_shape(0); ++k) {
        res({i, j}) += A({i, k}) * B({k, j});
      }
    }
  }
  return res;
}

}
