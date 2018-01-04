#include "gtest/gtest.h"

#include <gstorm.h>
#include <vector>
#include <iostream>
#include <random>
#include <functional>
#include <range/v3/all.hpp>

#include "experimental.h"

struct ViewTransform2 : public testing::Test {};

class TripleNum {
  public:
  constexpr TripleNum() {};
  int operator()(int a) const {
    return a*3;
  }
};

TEST_F(ViewTransform2, TestViewTransform2) {

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

  std::vector<int> vc(vsize);
  {
    gstorm::sycl_exec exec;

    auto ga = std::experimental::copy(exec, va);
    auto gb = std::experimental::copy(exec, vb);
    auto gc = std::experimental::copy(exec, vc);

    std::experimental::transform(exec, ranges::view::transform(ga, gb, std::plus<int>{}), gc, TripleNum{});
  }

  auto expected = ranges::view::transform(va, vb, std::plus<int>{})
                | ranges::view::transform(TripleNum{});

  EXPECT_TRUE(ranges::equal(expected, vc));
}
