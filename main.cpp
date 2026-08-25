#include <Shape.h>
#include <Tensor.h>
#include <TensorMath.h>
#include <TensorStats.h>
#include <TensorRandom.h>
#include <TensorLA.h>
#include <TensorBroadcast.h>
#include <iostream>

int main() {
  nerd::Tensor<double> A(nerd::Shape{3}, 1);
  nerd::Tensor<double> B(nerd::Shape{3, 3}, 1);
   
  for (int i = 0; i < A.size(); i++) {
    A[i] = i;
  }

  for (int i = 0; i < B.size(); i++) {
    B[i] = i;
  }

  std::cout << nerd::compatible(A.get_shape(), B.get_shape()) << std::endl;
  std::cout << nerd::output_shape(A.get_shape(), B.get_shape()) << std::endl;
  // auto C = nerd::matmult(A, B);
  // std::cout << C << std::endl;
  return 0;
}
