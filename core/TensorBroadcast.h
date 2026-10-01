#pragma once

#include <Tensor.h>

namespace nerd {

inline std::size_t max_rank(const Shape& A, const Shape& B) {
  return std::max(A.ndim(), B.ndim());
}

inline Shape align_dim(const Shape& A, std::size_t rank) {
  Shape A_align = A;
  while (A_align.ndim() < rank) {
    A_align.add(0,1);
  }
  return A_align;
}

inline bool compatible(const Shape& A, const Shape& B) {
  if (A == B) return true;

  std::size_t rank = max_rank(A, B);
  Shape A_align = align_dim(A, rank);
  Shape B_align = align_dim(B, rank);
  for (std::size_t i = 0; i < rank; ++i) {
    if (A_align[i] != B_align[i] && A_align[i] != 1 && B_align[i] != 1) {
      return false;
    }
  }
  
  return true;
}

inline Shape output_shape(const Shape& A, const Shape& B) {
  std::size_t rank = max_rank(A, B);
  Shape A_align = align_dim(A, rank);
  Shape B_align = align_dim(B, rank);

  Shape output(rank);
  for (std::size_t i = 0; i < rank; ++i) {
    if (A_align[i] > B_align[i]) {
      output[i] = A_align[i];
    } else {
      output[i] = B_align[i];
    }
  }

  return output;
}

// creating zero strides for broadcasting tensor
inline Shape broadcast_strides(const Shape& shape, const Shape& strides, const Shape& output_shape) {
  std::size_t rank = output_shape.ndim();
  Shape aligned_shape = align_dim(shape, rank);
  Shape effective_stride = strides;

  while (effective_stride.ndim() < rank) {
    effective_stride.add(0, 0);
  }

  for (std::size_t i = 0; i < rank; ++i) {
    if (aligned_shape[i] == 1 && output_shape[i] != 1) {
      effective_stride[i] = 0;
    }
  }

  return effective_stride;
}

// creating the offset for broadcasting operations (dot product of the current index and strides)
inline std::size_t broadcast_offset(const std::vector<std::size_t> coords, const Shape& strides) {
  std::size_t index = 0;

  for (std::size_t i = 0; i < coords.size(); ++i) {
    index += coords[i] * strides[i];
  }

  return index;
}

// binary operations for our tensor broadcasting
template <typename Type, typename BinaryOperation>
Tensor<Type> broadcast_binary(const Tensor<Type>& A, const Tensor<Type>& B, BinaryOperation operation) {
  Shape A_shape = A.get_shape();
  Shape B_shape = B.get_shape();
  assert(compatible(A_shape, B_shape));
  
  Shape output = output_shape(A_shape, B_shape);

  Tensor<Type> result(output);

  Shape A_strides = broadcast_strides(A_shape, A.get_strides(), output);

  Shape B_strides = broadcast_strides(B_shape, B.get_strides(), output);

  for (std::size_t i = 0; i < result.size(); ++i) {
    auto coords = result.unravel(i);

    std::size_t A_index = broadcast_offset(coords, A_strides);
    std::size_t B_index = broadcast_offset(coords, B_strides);

    result[i] = operation(A[A_index], B[B_index]);
  }

  return result;
}

}
