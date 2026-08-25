#include <Tensor.h>
#include <iostream>

void broadcast_test1() {
  // valid case
  nerd::Tensor<int> A({1,2,3,4});
  nerd::Tensor<int> B({2,2,1,1});
  if (A.compatible_dim(B)) {
    std::cout << "A is compatible with B" << std::endl;
  } else {
    std::cout << "False" << std::endl;
  }

  // invalid case
  nerd::Tensor<int> C({1,2,3,4});
  nerd::Tensor<int> D({1,5,1,2});
  if (C.compatible_dim(D)) {
    std::cout << "C is compatible with D" << std::endl;
  } else {
    std::cout <<  "False" << std::endl;
  }
}

void broadcast_test2() {
  // 3x3 matrix and array of size 3
  nerd::Tensor<int> E(nerd::Shape{3,3});
  nerd::Tensor<int> F(nerd::Shape{3});
  if (E.compatible_dim(F)) {
    std::cout << "E is compatible with F" << std::endl;
  } else {
    std::cout << "False" << std::endl;
  }

  std::cout << E.output_dim(F) << std::endl;
}

int main() {
  broadcast_test1();
  broadcast_test2();
  return 0;
}
