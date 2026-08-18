#include <benchmark/benchmark.h>
#include <Tensor.h>
#include <TensorRandom.h>
#include <TensorLA.h>

static void BM_NerdMatmult(benchmark::State& state) {
    std::size_t N = state.range(0);
    
    // creating NxN shape
    nerd::Shape shape{N, N};
    
    // initializing uniform matrices of float
    auto A = nerd::unifloat(shape, -1, 1);
    auto B = nerd::unifloat(shape, -1, 1);
    
    // the actual matmult
    for (auto _ : state) {
        auto C = nerd::matmult(A, B);
        
        benchmark::DoNotOptimize(C.empty()); 
        benchmark::ClobberMemory();
    }
    
    // counting flops
    state.counters["FLOPS"] = benchmark::Counter(
        state.iterations() * 2.0 * N * N * N, 
        benchmark::Counter::kIsRate
    );
}

BENCHMARK(BM_NerdMatmult)->RangeMultiplier(2)->Range(16, 4096);

BENCHMARK_MAIN();
