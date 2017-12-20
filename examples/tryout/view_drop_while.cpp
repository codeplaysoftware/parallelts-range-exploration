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

  auto add3 = [](auto a) { return a + 3; };
  std::vector<int> vb(vsize);
  {
    gstorm::sycl_exec exec;

    auto gb = std::experimental::copy(exec, vb);

    auto ints = ranges::view::ints(0, 1024)
              | ranges::view::drop_while([](auto a) { return a < 512; })
              | ranges::view::transform(add3);
    std::experimental::transform(exec, ints, gb, TripleNum{});
  }

  auto expected = ranges::view::ints(0, 1024)
                | ranges::view::drop_while([](auto a) { return a < 512; })
                | ranges::view::transform(add3)
                | ranges::view::transform(TripleNum{});

  if (not ranges::equal(expected, vb)) {
    std::cout << "Mismatch between expected and actual result!\n";
    return 1;
  }

  std::cout << "All good!\n";
}
