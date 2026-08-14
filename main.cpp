#include <Shape.h>
#include <Tensor.h>
#include <TensorMath.h>
#include <TensorStats.h>
#include <TensorRandom.h>
#include <TensorLA.h>
#include <iostream>

int main() {
  nerd::Tensor<double> A(nerd::Shape{1,3}, 1);
  nerd::Tensor<double> B(nerd::Shape{2,2}, 1);
   
  for (int i = 0; i < A.size(); i++) {
    A[i] = i;
  }

  for (int i = 0; i < B.size(); i++) {
    B[i] = i;
  }

  std::cout << A.compatible_dim(B) << std::endl;
  // auto C = nerd::matmult(A, B);
  // std::cout << C << std::endl;
  return 0;
}
