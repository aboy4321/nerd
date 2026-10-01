#include <Tensor.h>
#include <TensorBroadcast.h>
#include <TensorOperations.h>

#include <cassert>
#include <cmath>
#include <iostream>

using nerd::Tensor;
using nerd::Shape;

// -------------------------------------------------------
// Helper functions
// -------------------------------------------------------

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
// Broadcasting compatibility
// -------------------------------------------------------

void test_compatible_shapes() {
    assert(nerd::compatible(
        Shape{3, 1},
        Shape{3, 4}
    ));

    assert(nerd::compatible(
        Shape{3},
        Shape{3, 3}
    ));

    assert(nerd::compatible(
        Shape{2, 1, 3},
        Shape{1, 4, 3}
    ));

    std::cout << "test_compatible_shapes passed\n";
}

void test_incompatible_shapes() {
    assert(!nerd::compatible(
        Shape{2, 3},
        Shape{4, 3}
    ));

    assert(!nerd::compatible(
        Shape{2, 3},
        Shape{2, 4}
    ));

    assert(!nerd::compatible(
        Shape{2, 3, 4},
        Shape{2, 5, 4}
    ));

    std::cout << "test_incompatible_shapes passed\n";
}

// -------------------------------------------------------
// Output shape
// -------------------------------------------------------

void test_output_shape() {
    assert(
        nerd::output_shape(
            Shape{3, 1},
            Shape{3, 4}
        ) == Shape{3, 4}
    );

    assert(
        nerd::output_shape(
            Shape{3},
            Shape{3, 3}
        ) == Shape{3, 3}
    );

    assert(
        nerd::output_shape(
            Shape{2, 1, 3},
            Shape{1, 4, 3}
        ) == Shape{2, 4, 3}
    );

    std::cout << "test_output_shape passed\n";
}

// -------------------------------------------------------
// Broadcast strides
// -------------------------------------------------------

void test_broadcast_strides() {
    {
        Shape shape{3};
        Shape strides{1};
        Shape output{3, 3};

        Shape result =
            nerd::broadcast_strides(
                shape,
                strides,
                output
            );

        assert(result == Shape{0, 1});
    }

    {
        Shape shape{3, 1};
        Shape strides{1, 1};
        Shape output{3, 4};

        Shape result =
            nerd::broadcast_strides(
                shape,
                strides,
                output
            );

        assert(result == Shape{1, 0});
    }

    std::cout << "test_broadcast_strides passed\n";
}

// -------------------------------------------------------
// Broadcast offset
// -------------------------------------------------------

void test_broadcast_offset() {
    Shape strides{1, 0};

    assert(
        nerd::broadcast_offset({0, 0}, strides) == 0
    );

    assert(
        nerd::broadcast_offset({0, 3}, strides) == 0
    );

    assert(
        nerd::broadcast_offset({1, 2}, strides) == 1
    );

    assert(
        nerd::broadcast_offset({2, 3}, strides) == 2
    );

    std::cout << "test_broadcast_offset passed\n";
}

// -------------------------------------------------------
// Addition: {3,1} + {3,4}
// -------------------------------------------------------

void test_broadcast_add_singleton_dimension() {
    Tensor<int> A(Shape{3, 1});
    Tensor<int> B(Shape{3, 4});

    A[0] = 10;
    A[1] = 20;
    A[2] = 30;

    for (std::size_t i = 0; i < B.size(); ++i) {
        B[i] = static_cast<int>(i + 1);
    }

    Tensor<int> result = A + B;

    Tensor<int> expected(Shape{3, 4});

    int values[] = {
        11, 12, 13, 14,
        25, 26, 27, 28,
        39, 40, 41, 42
    };

    for (std::size_t i = 0; i < expected.size(); ++i) {
        expected[i] = values[i];
    }

    assert_tensor_equal(result, expected);

    std::cout
        << "test_broadcast_add_singleton_dimension passed\n";
}

// -------------------------------------------------------
// Rank expansion: {3} + {3,3}
// -------------------------------------------------------

void test_broadcast_add_rank_expansion() {
    Tensor<int> A(Shape{3});
    Tensor<int> B(Shape{3, 3});

    A[0] = 10;
    A[1] = 20;
    A[2] = 30;

    for (std::size_t i = 0; i < B.size(); ++i) {
        B[i] = static_cast<int>(i);
    }

    Tensor<int> result = A + B;

    Tensor<int> expected(Shape{3, 3});

    int values[] = {
        10, 21, 32,
        13, 24, 35,
        16, 27, 38
    };

    for (std::size_t i = 0; i < expected.size(); ++i) {
        expected[i] = values[i];
    }

    assert_tensor_equal(result, expected);

    std::cout
        << "test_broadcast_add_rank_expansion passed\n";
}

// -------------------------------------------------------
// N-dimensional broadcasting
// {2,1,3} + {1,4,3}
// -------------------------------------------------------

void test_multidimensional_broadcast() {
    Tensor<int> A(Shape{2, 1, 3});
    Tensor<int> B(Shape{1, 4, 3});

    for (std::size_t i = 0; i < A.size(); ++i) {
        A[i] = static_cast<int>(i + 1);
    }

    for (std::size_t i = 0; i < B.size(); ++i) {
        B[i] = static_cast<int>((i + 1) * 10);
    }

    Tensor<int> result = A + B;

    assert(result.get_shape() == Shape{2, 4, 3});

    int expected_values[] = {
        11, 22, 33,
        41, 52, 63,
        71, 82, 93,
        101, 112, 123,

        14, 25, 36,
        44, 55, 66,
        74, 85, 96,
        104, 115, 126
    };

    for (std::size_t i = 0; i < result.size(); ++i) {
        assert(result[i] == expected_values[i]);
    }

    std::cout
        << "test_multidimensional_broadcast passed\n";
}

// -------------------------------------------------------
// All four binary operations
// -------------------------------------------------------

void test_all_binary_operations() {
    Tensor<double> A(Shape{2, 1});
    Tensor<double> B(Shape{2, 2});

    A[0] = 10.0;
    A[1] = 20.0;

    B[0] = 1.0;
    B[1] = 2.0;
    B[2] = 4.0;
    B[3] = 5.0;

    auto add = A + B;
    auto sub = A - B;
    auto mul = A * B;
    auto div = A / B;

    assert(add[0] == 11.0);
    assert(add[1] == 12.0);
    assert(add[2] == 24.0);
    assert(add[3] == 25.0);

    assert(sub[0] == 9.0);
    assert(sub[1] == 8.0);
    assert(sub[2] == 16.0);
    assert(sub[3] == 15.0);

    assert(mul[0] == 10.0);
    assert(mul[1] == 20.0);
    assert(mul[2] == 80.0);
    assert(mul[3] == 100.0);

    assert(std::abs(div[0] - 10.0) < 1e-9);
    assert(std::abs(div[1] - 5.0) < 1e-9);
    assert(std::abs(div[2] - 5.0) < 1e-9);
    assert(std::abs(div[3] - 4.0) < 1e-9);

    std::cout
        << "test_all_binary_operations passed\n";
}

// -------------------------------------------------------
// Main
// -------------------------------------------------------

int main() {
    test_compatible_shapes();
    test_incompatible_shapes();
    test_output_shape();
    test_broadcast_strides();
    test_broadcast_offset();

    test_broadcast_add_singleton_dimension();
    test_broadcast_add_rank_expansion();
    test_multidimensional_broadcast();
    test_all_binary_operations();

    std::cout << "\nAll broadcast tests passed.\n";

    return 0;
}
