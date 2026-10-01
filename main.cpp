#include <Shape.h>
#include <Tensor.h>
#include <TensorMath.h>
#include <TensorStats.h>
#include <TensorRandom.h>
#include <TensorLA.h>
#include <TensorBroadcast.h>
#include <TensorOperations.h>
#include <iostream>

int main() {
  nerd::Tensor<double> A(nerd::Shape{3, 1});
  nerd::Tensor<double> B(nerd::Shape{3, 4});

  A[0] = 10;
  A[1] = 20;
  A[2] = 30;

  for (std::size_t i = 0; i < B.size(); ++i) {
    B[i] = i + 1;
  }

  auto C = A + B;
  auto D = A - B;
  auto E = A * B;
  auto F = A / B;

  std::cout << "A + B:\n" << C << "\n\n";
  std::cout << "A - B:\n" << D << "\n\n";
  std::cout << "A * B:\n" << E << "\n\n";
  std::cout << "A / B:\n" << F << '\n';
  return 0;
}
