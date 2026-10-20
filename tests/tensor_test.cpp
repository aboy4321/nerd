#include <Tensor.h>

#include <cassert>
#include <iostream>

using nerd::Tensor;
using nerd::Shape;

void test_tensor_size() {
    Tensor<int> A(Shape{2, 3, 4});

    assert(A.size() == 24);
    assert(A.ndim() == 3);

    std::cout << "test_tensor_size passed\n";
}

void test_tensor_indexing() {
    Tensor<int> A(Shape{2, 3});

    for (std::size_t i = 0; i < A.size(); ++i) {
        A[i] = static_cast<int>(i);
    }

    assert(A(0, 0) == 0);
    assert(A(0, 1) == 1);
    assert(A(0, 2) == 2);

    assert(A(1, 0) == 3);
    assert(A(1, 1) == 4);
    assert(A(1, 2) == 5);

    std::cout << "test_tensor_indexing passed\n";
}

void test_reshape() {
    Tensor<int> A(Shape{2, 3});

    for (std::size_t i = 0; i < A.size(); ++i) {
        A[i] = static_cast<int>(i);
    }

    auto B = A.reshape(Shape{3, 2});

    assert(B.get_shape() == Shape{3, 2});
    assert(B.size() == 6);

    for (std::size_t i = 0; i < A.size(); ++i) {
        assert(A[i] == B[i]);
    }

    std::cout << "test_reshape passed\n";
}

void test_unsqueeze() {
    Tensor<int> A(Shape{3, 4});

    auto B = A.unsqueeze(1);

    assert(B.get_shape() == Shape{3, 1, 4});
    assert(B.size() == A.size());

    std::cout << "test_unsqueeze passed\n";
}

void test_squeeze() {
    Tensor<int> A(Shape{1, 3, 1, 4});

    auto B = A.squeeze();

    assert(B.get_shape() == Shape{3, 4});

    std::cout << "test_squeeze passed\n";
}

int main() {
    test_tensor_size();
    test_tensor_indexing();
    test_reshape();
    test_unsqueeze();
    test_squeeze();

    std::cout << "\nAll Tensor tests passed.\n";

    return 0;
}
