#include "gtest/gtest.h"

#include <gstorm.h>
#include <vector>
#include <iostream>
#include <tuple>
#include <random>
#include <type_traits>
#include <range/v3/all.hpp>

#include "experimental.h"

struct ViewZipWorkaroundCopy : public testing::Test {};

struct AddComponents {
  constexpr AddComponents() {};
  int operator()(const std::tuple<int, int>& tpl) const {
    return std::get<0>(tpl) + std::get<1>(tpl);
  }
};

template<typename... Ts>
struct MakeCopyTuple {
  constexpr MakeCopyTuple() {};
  auto operator()(Ts... args) const {
    return std::tuple<std::remove_reference_t<Ts>...>(std::forward<Ts>(args)...);
  }
};

TEST_F(ViewZipWorkaroundCopy, TestViewZipWorkaroundCopy) {

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

    auto zip = ranges::view::zip_with(MakeCopyTuple<int&, int&>{}, ga, gb);
    std::experimental::transform(exec, zip, gc, AddComponents{});
  }

  auto multiply_components = [](const auto& a) { return std::get<0>(a) + std::get<1>(a); };
  auto expected = ranges::view::zip(va, vb)
                | ranges::view::transform(multiply_components);

  EXPECT_TRUE(ranges::equal(expected, vc));
}
