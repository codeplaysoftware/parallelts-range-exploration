#include "gtest/gtest.h"

#include <gstorm.h>
#include <vector>
#include <iostream>
#include <functional>
#include <range/v3/all.hpp>

#include "experimental.h"

struct Sum : public testing::Test {};

TEST_F(Sum, TestSum) {

  size_t vsize = 1024;

  // Input to the SYCL device
  std::vector<int> va(vsize, 1);

  {
    gstorm::sycl_exec exec;

    auto ga = std::experimental::copy(exec, va);

    auto result = std::experimental::reduce(exec, ga, 0, std::plus<int>{});
    auto expected = ranges::accumulate(va, 0);

    EXPECT_EQ(result, expected);
  }
}
