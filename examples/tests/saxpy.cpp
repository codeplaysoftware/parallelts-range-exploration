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

struct Saxpy : public testing::Test {};

template<typename T>
struct AddComponents {
  constexpr AddComponents() {};
  T operator()(const std::tuple<T, cl::sycl::global_ptr<T>>& tpl) const {
    return std::get<0>(tpl) + *std::get<1>(tpl);
  }
};

TEST_F(Saxpy, TestSaxpy) {

  size_t vsize = 1024;

  std::default_random_engine generator;
  std::uniform_real_distribution<float> distribution(0.0,10.0);

  auto generate_float =
    [&generator, &distribution]() { return distribution(generator); };

  // Input to the SYCL device
  std::vector<float> x(vsize);
  std::vector<float> y(vsize);

  ranges::generate(x, generate_float);
  ranges::generate(y, generate_float);

  const float a = generate_float();

  std::vector<float> z(vsize);

  {
    gstorm::sycl_exec exec;

    auto gpu_x = std::experimental::copy(exec, x);
    auto gpu_y = std::experimental::copy(exec, y);
    auto gpu_z = std::experimental::copy(exec, z);

    auto multiply_components =
      [](const auto& tpl) { return std::get<0>(tpl) * *std::get<1>(tpl); };

    auto ax = my_zip(ranges::view::repeat(a), gpu_x)
            | ranges::view::transform(multiply_components);
    std::experimental::transform(exec, my_zip(ax, gpu_y), gpu_z, AddComponents<float>{});
  }

  auto expected = ranges::view::transform(x, y, [a](auto x, auto y) { return x*a + y; });

  EXPECT_TRUE(ranges::equal(expected, z));
}
