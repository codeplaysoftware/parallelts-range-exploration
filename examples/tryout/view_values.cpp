#include <gstorm.h>
#include <vector>
#include <iostream>
#include <utility>
#include <random>
#include <range/v3/all.hpp>

#include "experimental.h"

class TripleNum {
  public:
  constexpr TripleNum() {};
  int operator()(int a) const {
    return a*3;
  }
};

int main() {

  size_t vsize = 1024;

  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(0,10);

  auto generate_int_pair = [&generator, &distribution]() {
    return std::make_pair(distribution(generator), distribution(generator)); };

  // Input to the SYCL device
  std::vector<std::pair<int, int>> va(vsize);
  ranges::generate(va, generate_int_pair);

  auto add3 = [](auto a) { return a + 3; };
  std::vector<int> vb(vsize);
  {
    gstorm::sycl_exec exec;

    auto ga = std::experimental::copy(exec, va);
    auto gb = std::experimental::copy(exec, vb);

    auto values = ranges::view::values(ga)
                | ranges::view::transform(add3);
    std::experimental::transform(exec, values, gb, TripleNum{});
  }

  auto expected = ranges::view::values(va)
                | ranges::view::transform(add3)
                | ranges::view::transform(TripleNum{});

  if (not ranges::equal(expected, vb)) {
    std::cout << "Mismatch between expected and actual result!\n";
    return 1;
  }

  std::cout << "All good!\n";
}
