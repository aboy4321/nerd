#include "Shape.h"
#include <cassert>

using namespace nerd;

void test_shape() {
  Shape shape{2, 3, 4};
  assert(shape[0] == 2);
  assert(shape[1] == 3);
  assert(shape[2] == 4);
  assert(shape.rank() == 3);
  assert(shape.size() == 24);
  assert(!(shape.empty()));
  shape.add(0, 2);
  std::cout << shape << std::endl;
}

int main() {
  test_shape();
}
