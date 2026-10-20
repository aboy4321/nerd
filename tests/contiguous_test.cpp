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

void test_contiguous_after_permute() {
    Tensor<int> A(Shape{2, 3});

    for (std::size_t i = 0; i < A.size(); ++i) {
        A[i] = static_cast<int>(i + 1);
    }

    /*
        A =
        [
            [1, 2, 3],
            [4, 5, 6]
        ]

        shape   = {2, 3}
        strides = {3, 1}
    */

    auto B = A.permute({1, 0});

    assert(B.get_shape() == Shape{3, 2});
    assert(!B.is_contiguous());

    /*
        Logically B should be:

        [
            [1, 4],
            [2, 5],
            [3, 6]
        ]

        But its physical storage is still:
        [1, 2, 3, 4, 5, 6]
    */

    auto C = B.contiguous();

    assert(C.get_shape() == Shape{3, 2});
    assert(C.is_contiguous());

    int expected[] = {
        1, 4,
        2, 5,
        3, 6
    };

    for (std::size_t i = 0; i < C.size(); ++i) {
        assert(C[i] == expected[i]);
    }

    std::cout << "test_contiguous_after_permute passed\n";
}

void test_contiguous_already_contiguous() {
    Tensor<int> A(Shape{2, 3});

    for (std::size_t i = 0; i < A.size(); ++i) {
        A[i] = static_cast<int>(i + 1);
    }

    assert(A.is_contiguous());

    auto B = A.contiguous();

    assert(B.is_contiguous());
    assert(B.get_shape() == A.get_shape());

    for (std::size_t i = 0; i < A.size(); ++i) {
        assert(B[i] == A[i]);
    }

    std::cout << "test_contiguous_already_contiguous passed\n";
}

void test_contiguous_3d_permute() {
  
    Tensor<int> A(Shape{2, 2, 3});

    for (std::size_t i = 0; i < A.size(); ++i) {
        A[i] = static_cast<int>(i);
    }

    auto B = A.permute({1, 0, 2});

    assert(!B.is_contiguous());

    auto C = B.contiguous();

    assert(C.is_contiguous());
    assert(C.get_shape() == Shape{2, 2, 3});

    for (std::size_t i = 0; i < C.size(); ++i) {
        auto coords = C.unravel(i);

        // B and C should represent the same logical value
        assert(C[i] == B(coords));
    }

    std::cout << "test_contiguous_3d_permute passed\n";
}

void test_reshape() {
    Tensor<int> A(Shape{2, 3});

    for (std::size_t i = 0; i < A.size(); ++i) {
        A[i] = static_cast<int>(i + 1);
    }

    /*
        A =
        [
            [1, 2, 3],
            [4, 5, 6]
        ]
    */

    auto B = A.reshape(Shape{3, 2});

    assert(B.get_shape() == Shape{3, 2});
    assert(B.size() == 6);
    assert(B.is_contiguous());

    int expected[] = {
        1, 2,
        3, 4,
        5, 6
    };

    for (std::size_t i = 0; i < B.size(); ++i) {
        assert(B[i] == expected[i]);
    }

    std::cout << "test_reshape passed\n";
}

void test_reshape_noncontiguous() {
    Tensor<int> A(Shape{2, 3});

    for (std::size_t i = 0; i < A.size(); ++i) {
        A[i] = static_cast<int>(i + 1);
    }

    /*
        A =
        [
            [1, 2, 3],
            [4, 5, 6]
        ]
    */

    auto B = A.permute({1, 0});

    /*
        B logically =
        [
            [1, 4],
            [2, 5],
            [3, 6]
        ]

        Logical flat order:
        1, 4, 2, 5, 3, 6
    */

    assert(!B.is_contiguous());

    auto C = B.reshape(Shape{2, 3});

    assert(C.get_shape() == Shape{2, 3});
    assert(C.size() == 6);
    assert(C.is_contiguous());

    /*
        C should logically be:

        [
            [1, 4, 2],
            [5, 3, 6]
        ]
    */

    int expected[] = {
        1, 4, 2,
        5, 3, 6
    };

    for (std::size_t i = 0; i < C.size(); ++i) {
        assert(C[i] == expected[i]);
    }

    std::cout << "test_reshape_noncontiguous passed\n";
}

int main() {
  test_contiguous_tensor();
  test_reshape_contiguous();
  test_permute_noncontiguous();
  test_contiguous_3d_permute();
  test_contiguous_already_contiguous();
  test_contiguous_after_permute();
  test_reshape();
  test_reshape_noncontiguous();
  return 0;
}
