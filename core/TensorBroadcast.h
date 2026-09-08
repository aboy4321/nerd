#include <Tensor.h>

namespace nerd {

std::size_t max_rank(const Shape& A, const Shape& B) {
  return std::max(A.ndim(), B.ndim());
}

Shape align_dim(const Shape& A, std::size_t rank) {
  Shape A_align = A;
  while (A_align.ndim() < rank) {
    A_align.add(0,1);
  }
  return A_align;
}

bool compatible(const Shape& A, const Shape& B) {
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

Shape output_shape(const Shape& A, const Shape& B) {
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

Shape zero_stride(const Shape& stride, std::size_t rank) {
    Shape eff_stride = stride;
    while (eff_stride.ndim() < rank) {
        eff_stride.add(0,0);
    }
    
    return eff_stride;
}

template <typename Type>
void broadcast(const Tensor<Type>& A, const Tensor<Type>& B) {
  Shape shape_A = A.get_shape();
  Shape shape_B = B.get_shape();

  if (!compatible(shape_A, shape_B)) {
    return;
  }

  Tensor<Type> result(output_shape(shape_A, shape_B));
  if (shape_A.ndim() < shape_B.ndim()) {
    Shape stride = zero_stride();
  }
}

// Shape create_unequal;
// aligns array to some row and column start

}
