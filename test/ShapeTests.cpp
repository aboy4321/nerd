#include "Shape.h"
#include <cassert>

using namespace nerd;

void test_shape() {
  Shape shape{2, 3, 4};
  std::cout << shape << std::endl;
}

int main() {
  test_shape();
}
