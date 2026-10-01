#pragma once

#include <Shape.h>
#include <Tensor.h>
#include <TensorMath.h>
#include <limits>

namespace nerd {

template <typename Type>
Type sum(const Tensor<Type>& T) {
  assert(!T.empty());
  return std::accumulate(T.begin(), T.end(), Type{0}); 
}

template <typename Type>
Tensor<Type> sum(const Tensor<Type>& T, std::size_t dim) {
  assert(dim < T.ndim());
  Shape other = T.get_shape();
  other.remove_dim(dim);
  Tensor<Type> res = Tensor<Type>::zeros(other);
  for (std::size_t i = 0; i < T.size(); ++i) {
    auto coord = T.unravel(i);
    coord.erase(coord.begin() + dim);
    res(coord) += T[i];
  }
  return res;
}

template <typename Type>
double mean(const Tensor<Type>& T) {
  assert(!T.empty());
  return static_cast<double>(sum(T)) / T.size();
}

template <typename Type>
Tensor<double> mean(const Tensor<Type>& T, std::size_t dim) {
  assert(dim < T.ndim());
  Shape shape = T.get_shape();
  Tensor<double> res = sum(T, dim);
  res /= static_cast<double>(shape[dim]);
  return res;
}

template <typename Type>
double var(const Tensor<Type>& T) {
  assert(!T.empty());
  double _mean = mean(T);
  double sqrd_diff_sum = 0.0;
  for (const auto& x : T) {
    double diff = x - _mean;
    sqrd_diff_sum += diff * diff;
  }
  return sqrd_diff_sum / T.size();
}

template <typename Type>
Tensor<double> var(const Tensor<Type>& T, std::size_t dim) {
  assert(dim < T.ndim());
  Shape shape = T.get_shape();
  Tensor<double> _mean = mean(T, dim);
  Tensor<double> res = Tensor<Type>::zeros(_mean.get_shape());
  for (std::size_t i = 0; i < T.size(); ++i) {
    auto coord = T.unravel(i);
    coord.erase(coord.begin() + dim);
    double diff = T[i] - _mean(coord);
    res(coord) += diff * diff;
  }
  return res / shape[dim];
}

template <typename Type>
double stdev(const Tensor<Type>& T) {
  assert(!T.empty());
  return std::sqrt(var(T));
}

template <typename Type>
Tensor<double> stdev(const Tensor<Type>& T, std::size_t dim) {
  assert(dim < T.ndim());
  return nerd::sqrt(var(T, dim));
}

template <typename Type>
Type min(const Tensor<Type>& T) {
  assert(!T.empty());
  auto min_it = std::min_element(T.begin(), T.end());
  return *min_it;
}

template <typename Type>
Tensor<Type> min(const Tensor<Type>& T, std::size_t dim) {
  assert(dim < T.ndim());
  Shape other = T.get_shape();
  other.remove_dim(dim);
  Tensor<Type> res(other, std::numeric_limits<Type>::max());
  for (std::size_t i = 0; i < T.size(); ++i) {
    auto coord = T.unravel(i);
    coord.erase(coord.begin() + dim);
    Type minimum = std::min(T[i], res(coord));
    res(coord) = minimum;
  }
  return res;
}

template <typename Type>
Type max(const Tensor<Type>& T) {
  assert(!T.empty());
  auto max_it = std::max_element(T.begin(), T.end());
  return *max_it;
}

template <typename Type>
Tensor<Type> max(const Tensor<Type>& T, std::size_t dim) {
  assert(dim < T.ndim());
  Shape other = T.get_shape();
  other.remove_dim(dim);
  Tensor<Type> res(other, std::numeric_limits<Type>::lowest());
  for (std::size_t i = 0; i < T.size(); ++i) {
    auto coord = T.unravel(i);
    coord.erase(coord.begin() + dim);
    Type maximum = std::max(T[i], res(coord));
    res(coord) = maximum;
  }
  return res;
}
}
