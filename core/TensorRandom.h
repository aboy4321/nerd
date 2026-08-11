#include <Tensor.h>
#include <TensorStats.h>
#include <random>
#include <vector>

std::random_device rd;
std::mt19937 gen(rd());

namespace nerd {

Tensor<double> unifloat(Shape shape, const double low = 0.0, const double high = 1.0)  {
  Tensor<double> res(shape);
  std::uniform_real_distribution<double> dist(low, high);
  for (double& x : res) {
    x = dist(gen);
  }
  return res;
}

Tensor<int> unifint(Shape shape, const int low = 0, const int high = 1)  {
  Tensor<int> res(shape);
  std::uniform_int_distribution<int> dist(low, high);
  for (int& x : res) {
    x = dist(gen);
  }
  return res;
}

Tensor<double> norm(Shape shape, const float mean = 0, const float stdev = 1) {
  Tensor <double> res(shape);
  std::normal_distribution<double> dist(mean, stdev);
  for (double& x : res) {
    x = dist(gen);
  }
  return res;
}

}
