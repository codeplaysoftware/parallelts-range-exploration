#include "gtest/gtest.h"

#include <functional>
#include <iostream>
#include <random>
#include <vector>

#include <gstorm.h>
#include <range/v3/all.hpp>

#include "experimental.h"

struct Reduce : public testing::TestWithParam<std::size_t> {};

TEST_P(Reduce, TestSum) {
  const auto vsize = GetParam();

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

TEST_P(Reduce, TestMult) {
  const auto vsize = GetParam();

  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(0, 1);

  // Input to the SYCL device
  std::vector<float> va(vsize);

  auto generate_float = [&generator, &distribution]() -> float {
    if (distribution(generator) == 1) {
      return 1.0 / 0.9;
    } else {
      return 0.9;
    }
  };

  ranges::generate(va, generate_float);

  {
    gstorm::sycl_exec exec;

    auto ga = std::experimental::copy(exec, va);

    auto result =
        std::experimental::reduce(exec, ga, 1.0f, std::multiplies<float>{});
    auto expected = ranges::accumulate(va, 1.0f, std::multiplies<float>{});

    // Is within 0.1% of expected?
    EXPECT_LT(std::abs(result - expected),
              0.001 * std::max(std::abs(result), std::abs(expected)));
  }
}

INSTANTIATE_TEST_CASE_P(TestSizes, Reduce,
                        testing::Values(16, 555, 1024, 32768, 32800, 33333));
