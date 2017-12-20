#include <gstorm.h>
#include <vector>
#include <iostream>
#include <range/v3/all.hpp>
#include <utility>
#include <random>

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

  auto generate_int_pair
    = [&]() { return std::make_pair(distribution(generator), distribution(generator)); };

  // Input to the SYCL device
  std::vector<std::pair<int, int>> va(vsize);
  ranges::generate(va, generate_int_pair);

  std::vector<int> vb(vsize);
  {
    gstorm::sycl_exec exec;

    auto ga = std::experimental::copy(exec, va);
    auto gb = std::experimental::copy(exec, vb);

    std::experimental::transform(exec, ranges::view::keys(ga), gb, TripleNum{});
  }

  auto expected = ranges::view::keys(va)
                | ranges::view::transform(TripleNum{});

  if (not ranges::equal(expected, vb)) {
    std::cout << "Mismatch between expected and actual result!\n";
    return 1;
  }

  std::cout << "All good!\n";
}
