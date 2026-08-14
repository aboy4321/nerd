#include <Tensor.h>
#include <iostream>

void broadcast_test() {
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

int main() {
  broadcast_test();
  return 0;
}
