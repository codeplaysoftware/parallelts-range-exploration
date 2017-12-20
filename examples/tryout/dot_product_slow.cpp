#include <gstorm.h>
#include <vector>
#include <functional>
#include <tuple>
#include <random>

#include <CL/sycl.hpp>
#include <range/v3/all.hpp>

#include "experimental.h"
#include "my_zip.h"

struct MultiplyComponents {
  constexpr MultiplyComponents() {};
  int operator()(std::tuple<cl::sycl::global_ptr<int>, cl::sycl::global_ptr<int>> a) const {
    return *std::get<0>(a) * *std::get<1>(a);
  }
};

int main() {

  size_t vsize = 1024;

  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(0,10);

  auto generate_int =
    [&generator, &distribution]() { return distribution(generator); };

  // Input to the SYCL device
  std::vector<int> va(vsize);
  std::vector<int> vb(vsize);
  std::vector<int> vtmp(vsize);

  ranges::generate(va, generate_int);
  ranges::generate(vb, generate_int);

  {
    gstorm::sycl_exec exec;

    auto ga = std::experimental::copy(exec, va);
    auto gb = std::experimental::copy(exec, vb);
    auto gtmp = std::experimental::copy(exec, vtmp);

    auto zipped = my_zip(ga, gb);
    std::experimental::transform(exec, zipped, gtmp, MultiplyComponents{});

    auto result = std::experimental::reduce(exec, gtmp, 0, std::plus<int>{});

    auto expected = ranges::accumulate(
        ranges::view::transform(va, vb, std::multiplies<int>{}), 0, std::plus<int>{});

    if (expected != result) {
      std::cout << "Mismatch between expected and actual result!\n";
      return 1;
    }
  }

  std::cout << "All good!\n";
}
