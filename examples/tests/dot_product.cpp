#include "gtest/gtest.h"

#include <gstorm.h>
#include <vector>
#include <iostream>
#include <functional>
#include <random>
#include <range/v3/all.hpp>

#include "experimental.h"

struct DotProduct : public testing::Test {};

TEST_F(DotProduct, TestDotProduct) {

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

    auto multiplied = ranges::view::transform(ga, gb, std::multiplies<int>{});
    auto result = std::experimental::reduce(exec, multiplied, 0, std::plus<int>{});

    auto expected = ranges::accumulate(ranges::view::transform(va, vb, std::multiplies<int>{}), 0, std::plus<int>{});

    EXPECT_EQ(result, expected);
  }
}
