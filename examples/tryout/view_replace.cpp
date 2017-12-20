#include <gstorm.h>
#include <vector>
#include <iostream>
#include <random>
#include <range/v3/all.hpp>

#include "experimental.h"

class Id {
  public:
  int operator()(int a) const {
    return a;
  }
};

int main() {

  size_t vsize = 1024;

  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(0,10);

  auto generate_int =
    [&generator, &distribution]() { return distribution(generator); };

  std::vector<int> va(vsize);
  ranges::generate(va, generate_int);

  std::vector<int> vb(vsize);
  {
    gstorm::sycl_exec exec;

    auto ga = std::experimental::copy(exec, va);
    auto gb = std::experimental::copy(exec, vb);

    std::experimental::transform(exec, ranges::view::replace(ga, 3, 33), gb, Id{});
  }

  auto expected = ranges::view::replace(va, 3, 33);

  if (not ranges::equal(expected, vb)) {
    std::cout << "Mismatch between expected and actual result!\n";
    return 1;
  }

  std::cout << "All good!\n";
}
