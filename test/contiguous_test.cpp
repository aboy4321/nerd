#include <Tensor.h>
#include <iostream>

using namespace nerd;


void test_contiguous_tensor() {
    Tensor<int> A(Shape{2, 3, 4});

    assert(A.is_contiguous());

    std::cout << "test_contiguous_tensor passed\n";
}

void test_reshape_contiguous() {
    Tensor<int> A(Shape{2, 3, 4});

    auto B = A.reshape(Shape{6, 4});

    assert(B.is_contiguous());

    std::cout << "test_reshape_contiguous passed\n";
}

void test_permute_noncontiguous() {
    Tensor<int> A(Shape{2, 3, 4});

    auto B = A.permute({1, 0, 2});

    assert(!B.is_contiguous());

    std::cout << "test_permute_noncontiguous passed\n";
}

int main() {
  test_contiguous_tensor();
  test_reshape_contiguous();
  test_permute_noncontiguous();
  return 0;
}
