#pragma once

#include <Tensor.h>
#include <cmath>

namespace nerd {

// for getting the square root of an entire Tensor
template <typename Type>
Tensor<double> sqrt(const Tensor<Type>& M) {
  Tensor<Type> input = M.contiguous();  

  Tensor<double> res(input.get_shape());
  for (std::size_t i = 0; i < input.size(); ++i) {
    res[i] = std::sqrt(input[i]);
  }

  return res;
}

template <typename Type>  
Tensor<double> exp(const Tensor<Type>& M) {
  Tensor<Type> input = M.contiguous();  

  Tensor<double> res(input.get_shape());
  for (std::size_t i = 0; i < input.size(); ++i) {
    res[i] = std::exp(input[i]);
  }
  return res;
}

template <typename Type, typename Scalar>
Tensor<double> pow(const Tensor<Type>& M, Scalar n) {
  Tensor<Type> input = M.contiguous();  

  Tensor<double> res(input.get_shape());
  for (std::size_t i = 0;  i < input.size(); ++i) {
    res[i] = std::pow(input[i], n);
  }
  return res;  
}


template <typename Type>  
Tensor<double> log(const Tensor<Type>& M) {
  Tensor<Type> input = M.contiguous();  

  Tensor<double> res(input.get_shape());
  for (std::size_t i = 0;  i < input.size(); ++i) {
    res[i] = std::log(input[i]);
  }
  return res;
}

template <typename Type>  
Tensor<Type> abs(const Tensor<Type>& M) {
  Tensor<Type> input = M.contiguous();  

  Tensor<Type> res(input.get_shape());
  for (std::size_t i = 0;  i < input.size(); ++i) {
    res[i] = std::abs(input[i]);
  }
  return res;
}

// trig functions, returns in radians
template <typename Type>
Tensor<double> sin(const Tensor<Type>& M) {
  Tensor<Type> input = M.contiguous();  

  Tensor<double> res(input.get_shape());
  for (std::size_t i = 0;  i < input.size(); ++i) {
    res[i] = std::sin(input[i]);
  }
  return res;
}

template <typename Type>
Tensor<double> cos(const Tensor<Type>& M) {
  Tensor<Type> input = M.contiguous();  

  Tensor<double> res(input.get_shape());
  for (std::size_t i = 0;  i < input.size(); ++i) {
    res[i] = std::cos(input[i]);
  }
  return res;
}

template <typename Type>
Tensor<double> tan(const Tensor<Type>& M) {
  Tensor<Type> input = M.contiguous();  

  Tensor<double> res(input.get_shape());
  for (std::size_t i = 0;  i < input.size(); ++i) {
    res[i] = std::tan(input[i]);
  }
  return res;
}
}
