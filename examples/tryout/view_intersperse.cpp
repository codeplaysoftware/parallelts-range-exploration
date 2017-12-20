#include <gstorm.h>
#include <vector>
#include <iostream>
#include <random>
#include <range/v3/all.hpp>

#include "experimental.h"

class Id {
  public:
  int operator()(char a) const {
    return a;
  }
};

int main() {

  size_t vsize = 1023;

  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(0,10);

  auto generate_int =
    [&generator, &distribution]() { return distribution(generator); };

  // Input to the SYCL device
  std::vector<char> va(vsize);
  ranges::generate(va, generate_int);

  std::vector<char> vb(vsize*2);
  {
    gstorm::sycl_exec exec;

    auto ga = std::experimental::copy(exec, va);
    auto gb = std::experimental::copy(exec, vb);

    auto interspersed = ranges::view::tail(ranges::view::intersperse(ga, 2));
    std::experimental::transform(exec, interspersed, gb, Id{});

  }

  auto expected = ranges::view::tail(ranges::view::intersperse(va, 2));

  if (not ranges::equal(expected, vb)) {
    std::cout << "Mismatch between expected and actual result!\n";
    return 1;
  }

  std::cout << "All good!\n";
}
