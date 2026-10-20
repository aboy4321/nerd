#include <Tensor.h>
#include <TensorLA.h>

#include <cassert>
#include <cmath>
#include <iostream>

using nerd::Tensor;
using nerd::Shape;

template <typename Type>
void assert_tensor_equal(
    const Tensor<Type>& A,
    const Tensor<Type>& B
) {
    assert(A.get_shape() == B.get_shape());
    assert(A.size() == B.size());

    for (std::size_t i = 0; i < A.size(); ++i) {
        assert(A[i] == B[i]);
    }
}

void assert_tensor_close(
    const Tensor<double>& A,
    const Tensor<double>& B,
    double tolerance = 1e-9
) {
    assert(A.get_shape() == B.get_shape());
    assert(A.size() == B.size());

    for (std::size_t i = 0; i < A.size(); ++i) {
        assert(std::abs(A[i] - B[i]) < tolerance);
    }
}


// -------------------------------------------------------
// 2x2 matrix multiplication
// -------------------------------------------------------

void test_matmul_2x2() {
    Tensor<int> A(Shape{2, 2});
    Tensor<int> B(Shape{2, 2});

    A[0] = 1; A[1] = 2;
    A[2] = 3; A[3] = 4;

    B[0] = 5; B[1] = 6;
    B[2] = 7; B[3] = 8;

    auto C = nerd::matmult(A, B);

    Tensor<int> expected(Shape{2, 2});

    expected[0] = 19;
    expected[1] = 22;
    expected[2] = 43;
    expected[3] = 50;

    assert_tensor_equal(C, expected);

    std::cout << "test_matmul_2x2 passed\n";
}


// -------------------------------------------------------
// Rectangular matrices
// {2,3} x {3,2} -> {2,2}
// -------------------------------------------------------

void test_rectangular_matmul() {
    Tensor<int> A(Shape{2, 3});
    Tensor<int> B(Shape{3, 2});

    int A_values[] = {
        1, 2, 3,
        4, 5, 6
    };

    int B_values[] = {
        7,  8,
        9, 10,
        11, 12
    };

    for (std::size_t i = 0; i < A.size(); ++i)
        A[i] = A_values[i];

    for (std::size_t i = 0; i < B.size(); ++i)
        B[i] = B_values[i];

    auto C = nerd::matmult(A, B);

    Tensor<int> expected(Shape{2, 2});

    expected[0] = 58;
    expected[1] = 64;
    expected[2] = 139;
    expected[3] = 154;

    assert_tensor_equal(C, expected);

    std::cout << "test_rectangular_matmul passed\n";
}


// -------------------------------------------------------
// Identity
// -------------------------------------------------------

void test_identity_matmul() {
    Tensor<double> A(Shape{3, 3});

    for (std::size_t i = 0; i < A.size(); ++i)
        A[i] = static_cast<double>(i + 1);

    auto I = Tensor<double>::identity(3);

    auto left  = nerd::matmult(I, A);
    auto right = nerd::matmult(A, I);

    assert_tensor_close(left, A);
    assert_tensor_close(right, A);

    std::cout << "test_identity_matmul passed\n";
}


// -------------------------------------------------------
// Matrix times column vector
// {2,3} x {3,1}
// -------------------------------------------------------

void test_matrix_column_vector() {
    Tensor<int> A(Shape{2, 3});
    Tensor<int> B(Shape{3, 1});

    int A_values[] = {
        1, 2, 3,
        4, 5, 6
    };

    int B_values[] = {
        2,
        3,
        4
    };

    for (std::size_t i = 0; i < A.size(); ++i)
        A[i] = A_values[i];

    for (std::size_t i = 0; i < B.size(); ++i)
        B[i] = B_values[i];

    auto C = nerd::matmult(A, B);

    assert(C.get_shape() == Shape{2, 1});

    assert(C[0] == 20);
    assert(C[1] == 47);

    std::cout << "test_matrix_column_vector passed\n";
}


// -------------------------------------------------------
// Zero matrix
// -------------------------------------------------------

void test_zero_matmul() {
    Tensor<double> A(Shape{3, 4}, 5.0);
    Tensor<double> B(Shape{4, 2}, 0.0);

    auto C = nerd::matmult(A, B);

    assert(C.get_shape() == Shape{3, 2});

    for (std::size_t i = 0; i < C.size(); ++i) {
        assert(C[i] == 0.0);
    }

    std::cout << "test_zero_matmul passed\n";
}


int main() {
    test_matmul_2x2();
    test_rectangular_matmul();
    test_identity_matmul();
    test_matrix_column_vector();
    test_zero_matmul();

    std::cout << "\nAll matrix multiplication tests passed.\n";
}
