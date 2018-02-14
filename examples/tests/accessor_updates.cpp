#include "gtest/gtest.h"

#include <vector>
#include <tuple>

#include <gstorm.h>
#include <range/v3/all.hpp>

#include "experimental.h"

struct AccessorUpdate : public testing::Test {};

struct Identity {
  constexpr Identity() {};

  template <typename T>
  T operator()(T a) const {
    return a;
  }
};

struct TripleFirst {
  constexpr TripleFirst() {};

  template <typename T>
  auto operator()(const T& a) const {
    return std::get<0>(a) * 3;
  }
};

TEST_F(AccessorUpdate, Bug) {
  int vsize = 1024;

  std::vector<int> va(vsize, 1);
  std::vector<int> vb(vsize, 2);
  {
    gstorm::sycl_exec exec;

    auto ga = std::experimental::copy(exec, va);
    auto gb = std::experimental::copy(exec, vb);

    auto id = ranges::view::transform(ga, [](auto a) { return a; });

    std::experimental::transform(exec, ga, gb, Identity{});

    auto indices = ranges::view::iota(0, vsize);
    auto zipped = ranges::view::zip(id, indices);

    std::experimental::transform(exec, zipped, gb, TripleFirst{});
  }

  auto expected = va
                | ranges::view::transform([](auto a) { return a*3; });

  EXPECT_TRUE(ranges::equal(expected, vb));
}
