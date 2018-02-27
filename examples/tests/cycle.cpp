#include "gtest/gtest.h"

#include <gstorm.h>
#include <vector>

#include <range/v3/all.hpp>

#include "experimental.h"

struct Cycle : public testing::Test {};

struct Identity {
  constexpr Identity() {}

  template <typename T>
  T operator()(T a) const {
    return a;
  }
};

TEST_F(Cycle, SyclCycle) {

  // Input to the SYCL device
  std::vector<int> input{0,1,2,3,4,5,6,7};
  std::vector<int> output(input.size());

  {
    gstorm::sycl_exec exec;
    auto g = std::experimental::copy(exec, input);
    auto go = std::experimental::copy(exec, output);

    auto shifted = ranges::view::cycle(g)
                 | ranges::view::slice(4, 12);

    std::experimental::transform(exec, shifted, go, Identity{});
  }

  std::vector<int> expected{4,5,6,7,0,1,2,3};

  EXPECT_TRUE(ranges::equal(expected, output));
}


