#include <benchmark/benchmark.h>

#include <Tensor.h>
#include <TensorMath.h>
#include <TensorRandom.h>

using nerd::Tensor;
using nerd::Shape;

static void BM_Matmult(benchmark::State& state) {
    const size_t n = state.range(0);

    // Generate inputs once.
    auto A = nerd::unifloat(
        Shape{n, n}, -1.0, 1.0
    );

    auto B = nerd::unifloat(
        Shape{n, n}, -1.0, 1.0
    );

    for (auto _ : state) {
        auto C = nerd::matmult(A, B);

        benchmark::DoNotOptimize(C);
    }

    // 2N^3 floating-point operations for NxN matrix multiplication.
    double flops = 2.0 * n * n * n;

    state.counters["GFLOPS"] =
        benchmark::Counter(
            flops,
            benchmark::Counter::kIsIterationInvariantRate
        );
}

BENCHMARK(BM_Matmult)
    ->RangeMultiplier(2)
    ->Range(32, 1024);

BENCHMARK_MAIN();
