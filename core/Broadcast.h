#include <Tensor.h>

namespace nerd {
bool compatible_dim(const Tensor& other) const {
  if (same_shape(other)) return true;

  Shape other_shape = other.get_shape();

  other_shape = stretch(other_shape);
  for (std::size_t i = 0; i < shape.ndim(); ++i) {
    if (other_shape[i] != shape[i] && other_shape[i] != 1 && shape[i] != 1) return false;
  }

  return true;
}

Shape output_dim(const Tensor& other) const {
  assert(compatible_dim(other));
  Shape other_shape = stretch(other.shape);
  Shape res;
  for (std::size_t i = 0; i < shape.ndim(); ++i) {
    if (shape[i] >= other_shape[i]) {
      res[i] = shape[i];
    } else {
      res[i] = other_shape[i];
    }
  }
  return res;
}

// temporary function, will be removed later
Shape stretch(Shape& other) const {
  while (other.ndim() < shape.ndim()) {
    other.add(0, 1);
  }
  return other;
}

}
