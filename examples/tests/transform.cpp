#include "gtest/gtest.h"

#include <gstorm.h>
#include <iostream>
#include <random>
#include <vector>

#include <range/v3/all.hpp>

#include "experimental.h"

struct TransformAlgorithm : public testing::TestWithParam<std::size_t> {};

class TripleNum {
 public:
  int operator()(int a) const { return a * 3; }
};

TEST_P(TransformAlgorithm, RunWithSize) {
  const auto vsize = GetParam();

  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(0, 10);

  auto generate_int = [&generator, &distribution]() {
    return distribution(generator);
  };

  // Input to the SYCL device
  std::vector<int> va(vsize);
  ranges::generate(va, generate_int);

  std::vector<int> vb(vsize);
  {
    gstorm::sycl_exec exec;

    auto ga = std::experimental::copy(exec, va);
    auto gb = std::experimental::copy(exec, vb);

    std::experimental::transform(exec, ga, gb, TripleNum{});
  }

  auto expected = va | ranges::view::transform(TripleNum{});

  EXPECT_TRUE(ranges::equal(expected, vb));
}

INSTANTIATE_TEST_CASE_P(TestSizes, TransformAlgorithm,
                        testing::Values(16, 555, 1024));
