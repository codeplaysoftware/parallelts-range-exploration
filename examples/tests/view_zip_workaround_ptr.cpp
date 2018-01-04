#include "gtest/gtest.h"

#include <gstorm.h>
#include <vector>
#include <iostream>
#include <tuple>
#include <type_traits>
#include <random>

#include <range/v3/all.hpp>
#include <CL/sycl.hpp>

#include "experimental.h"

struct ViewZipWorkaroundPtr : public testing::Test {};

struct AddComponents {
  constexpr AddComponents() {};
  int operator()(const std::tuple<cl::sycl::global_ptr<int>, cl::sycl::global_ptr<int>>& tpl) const {
    return *std::get<0>(tpl) + *std::get<1>(tpl);
  }
};

template<typename... Ts>
struct MakePointerTuple {
  constexpr MakePointerTuple() {};
  auto operator()(Ts... args) const {
    return std::make_tuple(cl::sycl::global_ptr<std::remove_reference_t<Ts>>(&args)...);
  }
};

TEST_F(ViewZipWorkaroundPtr, TestViewZipWorkaroundPtr) {

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

    auto zip = ranges::view::zip_with(MakePointerTuple<int&, int&>{}, ga, gb);
    std::experimental::transform(exec, zip, gc, AddComponents{});
  }

  auto multiply_components = [](const auto& a) { return std::get<0>(a) + std::get<1>(a); };
  auto expected = ranges::view::zip(va, vb)
                | ranges::view::transform(multiply_components);

  EXPECT_TRUE(ranges::equal(expected, vc));
}
