#pragma once

#include <Tensor.h>
#include <TensorBroadcast.h>

namespace nerd {

// creating the binary helper functions for tensor broadcasting:
template <typename Type>
Tensor<Type> subtract(const Tensor<Type>& A, const Tensor<Type>& B) {
  return broadcast_binary(
      A,
      B,
      [](const Type& a, const Type& b) {return a - b;}
  );
}

template <typename Type>
Tensor<Type> add(const Tensor<Type>& A, const Tensor<Type>& B) {
  return broadcast_binary(
      A,
      B,
      [](const Type& a, const Type& b) {return a + b;}
  );
}

template <typename Type>
Tensor<Type> divide(const Tensor<Type>& A, const Tensor<Type>& B) {
  return broadcast_binary(
      A,
      B,
      [](const Type& a, const Type& b) {return a / b;}
  );
}

template <typename Type>
Tensor<Type> multiply(const Tensor<Type>& A, const Tensor<Type>& B) {
  return broadcast_binary(
      A,
      B,
      [](const Type& a, const Type& b) {return a * b;}
  );
}

// accessible operations for Tensors:
template <typename Type>
Tensor<Type> operator-(
    const Tensor<Type>& A,
    const Tensor<Type>& B
) {
    return subtract(A, B);
}

template <typename Type>
Tensor<Type> operator*(
    const Tensor<Type>& A,
    const Tensor<Type>& B
) {
    return multiply(A, B);
}

template <typename Type>
Tensor<Type> operator/(
    const Tensor<Type>& A,
    const Tensor<Type>& B
) {
    return divide(A, B);
}

template <typename Type>
Tensor<Type> operator+(
    const Tensor<Type>& A,
    const Tensor<Type>& B
) {
    return add(A, B);
}

}
