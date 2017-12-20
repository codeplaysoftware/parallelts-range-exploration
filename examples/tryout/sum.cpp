#include <gstorm.h>
#include <vector>
#include <iostream>
#include <functional>
#include <range/v3/all.hpp>

#include "experimental.h"

int main() {

  size_t vsize = 1024;

  // Input to the SYCL device
  std::vector<int> va(vsize, 1);

  {
    gstorm::sycl_exec exec;

    auto ga = std::experimental::copy(exec, va);

    auto result = std::experimental::reduce(exec, ga, 0, std::plus<int>{});
    auto expected = ranges::accumulate(va, 0);

    if (result != expected) {
      std::cout << "Mismatch between expected and actual result!\n";
      return 1;
    }
  }

  std::cout << "All good!\n";
}
