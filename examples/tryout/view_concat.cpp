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
  std::vector<int> vc(vsize*2);

  generate(va, generate_int);
  generate(vb, generate_int);

  {
    sycl_exec exec;

    auto ga = std::experimental::copy(exec, va);
    auto gb = std::experimental::copy(exec, vb);
    auto gc = std::experimental::copy(exec, vc);

    std::experimental::transform(exec, view::transform(view::concat(ga, gb), Add3{}), gb, TripleNum{});
  }

  std::vector<int> gold(vsize*2);

  transform(view::transform(view::concat(va, vb), Add3{}), gold.begin(), TripleNum{});

  for (size_t i = 0; i < vsize; i++) {
    if (gold[i] != vc[i]) {
      std::cout << "Mismatch at position " << i << "\n";
      return 1;
    }
  }

  std::cout << "All good!\n";
}
