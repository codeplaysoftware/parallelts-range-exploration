#include "gtest/gtest.h"

#include <gstorm.h>
#include <vector>
#include <iostream>
#include <functional>
#include <tuple>
#include <range/v3/all.hpp>

#include "experimental.h"
#include "my_zip.h"

struct DotProductZipCopy : public testing::Test {};

TEST_F(DotProductZipCopy, TestDotProductZipCopy) {

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

    auto id = [](int a) { return a; };
    auto multiply_components =
      [](const auto& a) { return std::get<0>(a) * std::get<1>(a); };

    auto a_id = ranges::view::transform(ga, id);
    auto b_id = ranges::view::transform(gb, id);

    auto multiplied = ranges::view::zip(a_id, b_id)
                    | ranges::view::transform(multiply_components);

    auto result = std::experimental::reduce(exec, multiplied, 0, std::plus<int>{});

    auto expected = ranges::accumulate(
          ranges::view::zip(va, vb)
        | ranges::view::transform(multiply_components), 0, std::plus<int>{});

    EXPECT_EQ(result, expected);
  }
}
