#include <Tensor.h>
#include <TensorBroadcast.h>
#include <iostream>

using namespace nerd;

void incompatible_shape_test() {
  std::cout << nerd::compatible(
      nerd::Shape{2, 3},
      nerd::Shape{2, 4}
  ) << '\n';

  std::cout << nerd::compatible(
      nerd::Shape{2, 3, 4},
      nerd::Shape{2, 5, 4}
  ) << '\n';

  std::cout << nerd::compatible(
      nerd::Shape{3},
      nerd::Shape{2, 4}
  ) << '\n';

}

int main() {
  incompatible_shape_test();
  return 0;
}
