#include <gstorm.h>
#include <vector>
#include <iostream>
#include <random>
#include <range/v3/all.hpp>

#include "experimental.h"

int main() {

  size_t vsize = 1024;

  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(0,10);

  auto generate_int =
    [&generator, &distribution]() { return distribution(generator); };

  // Input to the SYCL device
  std::vector<int> va(vsize);
  ranges::generate(va, generate_int);

  auto vb = ranges::view::ints(3, (int)vsize+3);

  auto expected = ranges::accumulate(
      ranges::view::transform(va, vb, std::multiplies<int>{}), 0, std::plus<int>{});

  {
    gstorm::sycl_exec exec;

    auto ga = std::experimental::copy(exec, va);

    auto multiplied = ranges::view::transform(ga, vb, std::multiplies<int>{});
    auto result = std::experimental::reduce(exec, multiplied, 0, std::plus<int>{});

    if (expected != result) {
      std::cout << "Mismatch between expected and actual result!\n";
      return 1;
    }
  }

  std::cout << "All good!\n";
}
