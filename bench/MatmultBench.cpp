#include <chrono>
#include <Tensor.h>
#include <TensorRandom.h>
#include <TensorLA.h>

void benchmark_matmult(size_t size, int iterations) {
    Tensor<float> A = unifloat({size, size}, -1, 1);
    <float> B = unifloat({size, size}, -1, 1);

    // Warm-up
    for (int i = 0; i < 5; ++i) {
        auto C = matmult(A, B);
    }

    auto start = high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        auto C = nerd::matmult(A, B);

        // Prevent the compiler from completely eliminating the work
        volatile float result = C(0, 0);
        (void)result;
    }

    auto end = high_resolution_clock::now();

    double total_time =
        duration<double>(end - start).count();

    double average_time =
        total_time / iterations;

    double operations =
        2.0 * size * size * size;

    double gflops =
        operations / average_time / 1e9;

    std::cout << size << "x" << size
         << " | "
         << average_time * 1000 << " ms"
         << " | "
         << gflops << " GFLOPS"
         << '\n';
}

int main() {
    vector<size_t> sizes = {
        32,
        64,
        128,
        256,
        512
    };

    std::cout << "Matrix Multiplication Benchmark\n";
    std::cout << "--------------------------------\n";

    for (std::size_t size : sizes) {
        int iterations;

        if (size <= 128)
            iterations = 100;
        else if (size <= 256)
            iterations = 25;
        else
            iterations = 5;

        benchmark_matmult(size, iterations);
    }

    return 0;
}


