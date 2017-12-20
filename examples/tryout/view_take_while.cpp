#include <gstorm.h>
#include <vector>
#include <iostream>
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

  // Input to the SYCL device
  std::vector<int> va(vsize);
  for (size_t i = 0; i < vsize; i++) {
    va[i] = i;
  }

  auto take_condition = [](auto a) { return a < 512; };
  auto add3 = [](auto a) { return a + 3; };
  std::vector<int> vb(vsize);
  {
    gstorm::sycl_exec exec;

    auto ga = std::experimental::copy(exec, va);
    auto gb = std::experimental::copy(exec, vb);

    auto taken = ga
               | ranges::view::take_while(take_condition)
               | ranges::view::transform(add3);

    std::experimental::transform(exec, taken, gb, TripleNum{});
  }

  auto expected = va
                | ranges::view::take_while(take_condition)
                | ranges::view::transform(add3)
                | ranges::view::transform(TripleNum{});

  if (not ranges::equal(expected, vb)) {
    std::cout << "Mismatch between expected and actual result!\n";
    return 1;
  }

  std::cout << "All good!\n";
}
