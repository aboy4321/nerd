#include <Tensor.h>
#include <TensorStats.h>

#include <cassert>
#include <cmath>
#include <iostream>

using nerd::Tensor;
using nerd::Shape;

constexpr double EPS = 1e-9;

bool close(double a, double b) {
    return std::abs(a - b) < EPS;
}


// -------------------------------------------------------
// Scalar sum
// -------------------------------------------------------

void test_sum() {
    Tensor<int> A(Shape{4});

    A[0] = 1;
    A[1] = 2;
    A[2] = 3;
    A[3] = 4;

    assert(nerd::sum(A) == 10);

    std::cout << "test_sum passed\n";
}


// -------------------------------------------------------
// Scalar mean
// -------------------------------------------------------

void test_mean_double() {
    Tensor<double> A(Shape{4});

    A[0] = 1.0;
    A[1] = 2.0;
    A[2] = 3.0;
    A[3] = 4.0;

    assert(close(nerd::mean(A), 2.5));

    std::cout << "test_mean_double passed\n";
}


// This test is especially important for integer tensors.
void test_mean_integer() {
    Tensor<int> A(Shape{2});

    A[0] = 1;
    A[1] = 2;

    assert(close(nerd::mean(A), 1.5));

    std::cout << "test_mean_integer passed\n";
}


// -------------------------------------------------------
// Scalar variance
// -------------------------------------------------------

void test_variance() {
    Tensor<double> A(Shape{4});

    A[0] = 1;
    A[1] = 2;
    A[2] = 3;
    A[3] = 4;

    // Population variance:
    //
    // mean = 2.5
    //
    // ((1-2.5)^2 + (2-2.5)^2
    //  + (3-2.5)^2 + (4-2.5)^2) / 4
    //
    // = 1.25

    assert(close(nerd::var(A), 1.25));

    std::cout << "test_variance passed\n";
}


// -------------------------------------------------------
// Scalar standard deviation
// -------------------------------------------------------

void test_stdev() {
    Tensor<double> A(Shape{4});

    A[0] = 1;
    A[1] = 2;
    A[2] = 3;
    A[3] = 4;

    assert(close(
        nerd::stdev(A),
        std::sqrt(1.25)
    ));

    std::cout << "test_stdev passed\n";
}


// -------------------------------------------------------
// Scalar min / max
// -------------------------------------------------------

void test_min_max() {
    Tensor<double> A(Shape{5});

    A[0] = 4.0;
    A[1] = -7.0;
    A[2] = 10.0;
    A[3] = 2.0;
    A[4] = -1.0;

    assert(close(nerd::min(A), -7.0));
    assert(close(nerd::max(A), 10.0));

    std::cout << "test_min_max passed\n";
}


// -------------------------------------------------------
// Sum across dimensions
//
// A =
// [
//   [1, 2, 3],
//   [4, 5, 6]
// ]
// -------------------------------------------------------

void test_sum_dimension() {
    Tensor<double> A(Shape{2, 3});

    for (std::size_t i = 0; i < A.size(); ++i) {
        A[i] = static_cast<double>(i + 1);
    }

    auto dim0 = nerd::sum(A, 0);

    assert(dim0.get_shape() == Shape{3});

    assert(close(dim0[0], 5.0));
    assert(close(dim0[1], 7.0));
    assert(close(dim0[2], 9.0));

    auto dim1 = nerd::sum(A, 1);

    assert(dim1.get_shape() == Shape{2});

    assert(close(dim1[0], 6.0));
    assert(close(dim1[1], 15.0));

    std::cout << "test_sum_dimension passed\n";
}


// -------------------------------------------------------
// Mean across dimensions
// -------------------------------------------------------

void test_mean_dimension() {
    Tensor<double> A(Shape{2, 3});

    for (std::size_t i = 0; i < A.size(); ++i) {
        A[i] = static_cast<double>(i + 1);
    }

    auto dim0 = nerd::mean(A, 0);

    assert(dim0.get_shape() == Shape{3});

    assert(close(dim0[0], 2.5));
    assert(close(dim0[1], 3.5));
    assert(close(dim0[2], 4.5));

    auto dim1 = nerd::mean(A, 1);

    assert(dim1.get_shape() == Shape{2});

    assert(close(dim1[0], 2.0));
    assert(close(dim1[1], 5.0));

    std::cout << "test_mean_dimension passed\n";
}


// -------------------------------------------------------
// Variance across dimensions
// -------------------------------------------------------

void test_variance_dimension() {
    Tensor<double> A(Shape{2, 3});

    A[0] = 1;
    A[1] = 2;
    A[2] = 3;

    A[3] = 4;
    A[4] = 5;
    A[5] = 6;

    auto dim1 = nerd::var(A, 1);

    assert(dim1.get_shape() == Shape{2});

    // var([1,2,3]) = 2/3
    // var([4,5,6]) = 2/3

    assert(close(dim1[0], 2.0 / 3.0));
    assert(close(dim1[1], 2.0 / 3.0));

    auto dim0 = nerd::var(A, 0);

    assert(dim0.get_shape() == Shape{3});

    // Each column consists of values differing by 3:
    // [1,4], [2,5], [3,6]
    //
    // Population variance = 2.25

    assert(close(dim0[0], 2.25));
    assert(close(dim0[1], 2.25));
    assert(close(dim0[2], 2.25));

    std::cout << "test_variance_dimension passed\n";
}


// -------------------------------------------------------
// Standard deviation across dimension
// -------------------------------------------------------

void test_stdev_dimension() {
    Tensor<double> A(Shape{2, 3});

    for (std::size_t i = 0; i < A.size(); ++i) {
        A[i] = static_cast<double>(i + 1);
    }

    auto result = nerd::stdev(A, 1);

    double expected = std::sqrt(2.0 / 3.0);

    assert(close(result[0], expected));
    assert(close(result[1], expected));

    std::cout << "test_stdev_dimension passed\n";
}


// -------------------------------------------------------
// Min across dimensions
// -------------------------------------------------------

void test_min_dimension() {
    Tensor<double> A(Shape{2, 3});

    double values[] = {
         5, -2,  9,
         1,  7, -4
    };

    for (std::size_t i = 0; i < A.size(); ++i)
        A[i] = values[i];

    auto dim0 = nerd::min(A, 0);

    assert(close(dim0[0], 1));
    assert(close(dim0[1], -2));
    assert(close(dim0[2], -4));

    auto dim1 = nerd::min(A, 1);

    assert(close(dim1[0], -2));
    assert(close(dim1[1], -4));

    std::cout << "test_min_dimension passed\n";
}


// -------------------------------------------------------
// Max across dimensions
//
// Use ALL NEGATIVE numbers intentionally.
// -------------------------------------------------------

void test_max_dimension_negative_values() {
    Tensor<double> A(Shape{2, 3});

    double values[] = {
        -5, -2, -9,
        -1, -7, -4
    };

    for (std::size_t i = 0; i < A.size(); ++i)
        A[i] = values[i];

    auto dim0 = nerd::max(A, 0);

    assert(close(dim0[0], -1));
    assert(close(dim0[1], -2));
    assert(close(dim0[2], -4));

    auto dim1 = nerd::max(A, 1);

    assert(close(dim1[0], -2));
    assert(close(dim1[1], -1));

    std::cout
        << "test_max_dimension_negative_values passed\n";
}


// -------------------------------------------------------
// Main
// -------------------------------------------------------

int main() {
    test_sum();

    test_mean_double();
    test_mean_integer();

    test_variance();
    test_stdev();

    test_min_max();

    test_sum_dimension();
    test_mean_dimension();
    test_variance_dimension();
    test_stdev_dimension();

    test_min_dimension();
    test_max_dimension_negative_values();

    std::cout << "\nAll statistics tests passed.\n";

    return 0;
}
