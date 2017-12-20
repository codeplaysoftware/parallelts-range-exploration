#include <gstorm.h>
#include <vector>
#include <iostream>
#include <functional>
#include <tuple>
#include <random>

#include <range/v3/all.hpp>

#include "experimental.h"
#include "my_zip.h"

int main() {

  size_t vsize = 1024;

  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(0,10);

  auto generate_int =
    [&generator, &distribution]() { return distribution(generator); };

  // Input to the SYCL device
  std::vector<int> va(vsize);
  std::vector<int> vb(vsize);

  ranges::generate(va, generate_int);
  ranges::generate(vb, generate_int);

  {
    gstorm::sycl_exec exec;

    auto ga = std::experimental::copy(exec, va);
    auto gb = std::experimental::copy(exec, vb);

    auto multiplied = my_zip(ga, gb) | ranges::view::transform(
        [](const auto& tpl) { return *std::get<0>(tpl) * *std::get<1>(tpl); });

    auto result = std::experimental::reduce(exec, multiplied, 0, std::plus<int>{});

    auto multiply_components =
      [](const auto& a) { return std::get<0>(a) * std::get<1>(a); };
    auto expected = ranges::accumulate(
          ranges::view::zip(va, vb)
        | ranges::view::transform(multiply_components), 0, std::plus<int>{});

    if (expected != result) {
      std::cout << "Mismatch between expected and actual result!\n";
      return 1;
    }
  }

  std::cout << "All good!\n";
}
