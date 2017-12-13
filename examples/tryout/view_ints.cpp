#include <gstorm.h>
#include <vector>
#include <iostream>
#include <memory>
#include <range/v3/all.hpp>
#include <type_traits>

#include <CL/sycl.hpp>

#include "experimental.h"

using namespace gstorm;
using namespace cl::sycl;
using namespace ranges::v3;

struct Add {
  constexpr Add() {};
  int operator()(int a, int b) const {
    return a + b;
  }
};

struct Mult {
  constexpr Mult() {};
  int operator()(int a, int b) const {
    return a * b;
  }
};

int main() {

  size_t vsize = 1024;

  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(0,10);

  auto generate_int = [&]() { return distribution(generator); };

  std::vector<int> va(vsize);

  generate(va, generate_int);

  auto vb = view::ints(3, (int)vsize+3);

  auto gold = accumulate(view::transform(va, vb, Mult{}), 0, Add{});

  {
    sycl_exec exec;

    auto ga = std::experimental::copy(exec, va);
    // auto gb = std::experimental::copy(exec, vb);

    auto multiplied = view::transform(ga, vb, Mult{});
    auto result = std::experimental::reduce(exec, multiplied, 0, Add{});

    if (gold != result) {
      std::cout << "Mismatch!\n";
      return 1;
    }
  }

  std::cout << "All good!\n";
}
