#include <Tensor.h>
#include <TensorBroadcast.h>
#include <iostream>

void broadcast_test1() {
  // valid case
  nerd::Tensor<int> A({1, 2, 3, 4});
  nerd::Tensor<int> B({2, 2, 1, 1});
  if (nerd::compatible(A.get_shape(), B.get_shape())) {
    std::cout << "A is compatible with B" << std::endl;
  } else {
    std::cout << "False" << std::endl;
  }

  // invalid case
  nerd::Tensor<int> A({1, 2, 3, 4});
  nerd::Tensor<int> B({1, 5, 1, 2});
  if (nerd::compatible(A.get_shape(), B.get_shape())) {
    std::cout << "A is compatible with B" << std::endl;
  } else {
    std::cout <<  "False" << std::endl;
  }
}

void broadcast_test2() {
  // 3x3 matrix and array of size 3
  nerd::Tensor<int> A(nerd::Shape{3, 3});
  nerd::Tensor<int> B(nerd::Shape{3});
  if (nerd::compatible(A.get_shape(), B.get_shape())) {
    std::cout << "A is compatible with B" << std::endl;
  } else {
    std::cout << "False" << std::endl;
  }

  std::cout << nerd::output_shape(A.get_shape(), B.get_shape()) << std::endl;
}

void broadcast_test3() {
    nerd::Tensor<int> A(nerd::Shape{2, 3, 6, 2});
    nerd::Tensor<int> B(nerd::Shape{6, 2});
    if (nerd::compatible(A.get_shape(), B.get_shape())) {
      std::cout << "A is compatible with B" << std::endl;
    } else {
      std::cout << "False" << std::endl;
    }

}

int main() {
  broadcast_test1();
  broadcast_test2();
  broadcast_test3();
  return 0;
}
