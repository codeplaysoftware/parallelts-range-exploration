#include "gtest/gtest.h"

#include <gstorm.h>
#include <vector>
#include <iostream>
#include <tuple>
#include <random>

#include <range/v3/all.hpp>
#include <CL/sycl.hpp>

#include "experimental.h"
#include "my_zip.h"

struct MyZipGvectorView : public testing::Test {};

struct AddComponents {
  constexpr AddComponents() {};
  int operator()(const std::tuple<int, cl::sycl::global_ptr<int>>& tpl) const {
    return std::get<0>(tpl) + *std::get<1>(tpl);
  }
};

TEST_F(MyZipGvectorView, TestMyZipGvectorView) {

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

    auto added_one = ranges::view::transform(ga, [](auto a) { return a + 1; });
    auto zip = my_zip(added_one, gb);
    std::experimental::transform(exec, zip, gc, AddComponents{});
  }

  auto expected = ranges::view::zip(
      va | ranges::view::transform([](auto a) { return a + 1; }),
      vb)
    | ranges::view::transform([](const auto& tpl) { return std::get<0>(tpl) + std::get<1>(tpl); });

  EXPECT_TRUE(ranges::equal(expected, vc));
}
