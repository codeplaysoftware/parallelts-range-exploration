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

class TripleNum {
  public:
  constexpr TripleNum() {};
  int operator()(int a) const {
    return a*3;
  }
};

class Add3 {
  public:
  constexpr Add3() {};
  int operator()(int a) const {
    return a+3;
  }
};

int main() {

  size_t vsize = 1024;

  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(0,10);

  auto generate_int = [&]() { return distribution(generator); };

  std::vector<int> va(vsize);
  std::vector<int> vb(vsize);

  generate(va, generate_int);

  {
    sycl_exec exec;

    auto ga = std::experimental::copy(exec, va);
    auto gb = std::experimental::copy(exec, vb);

    std::experimental::transform(exec, view::transform(view::stride(ga, 2), Add3{}), gb, TripleNum{});
  }

  std::vector<int> gold(vsize);

  transform(view::transform(view::stride(va, 2), Add3{}), gold.begin(), TripleNum{});

  for (size_t i = 0; i < vsize; i++) {
    if (gold[i] != vb[i]) {
      std::cout << "Mismatch at position " << i << "\n";
      return 1;
    }
  }

  std::cout << "All good!\n";
}
