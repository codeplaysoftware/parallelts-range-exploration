#include "gtest/gtest.h"

#include <gstorm.h>
#include <random>
#include <vector>
#include <utility>

#include <CL/sycl.hpp>
#include <range/v3/all.hpp>

#include "experimental.h"
#include "my_zip.h"

struct Sgemv : public testing::Test {};

struct DotProduct {
  constexpr DotProduct(){};
  template <typename T>
  float operator()(const T& tpl) const {
    auto row = std::get<0>(tpl);
    auto x = std::get<1>(tpl);

    return ranges::inner_product(row, x, 0.0f);
  }
};

struct MultiplyComponents {
  constexpr MultiplyComponents() {};

  template <typename T>
  int operator()(const T& a) const {
    return std::get<0>(a) * std::get<1>(a);
  }
};

TEST_F(Sgemv, MatrixVectorMultplication) {
  const size_t vsize = 1024;

  std::default_random_engine generator;
  std::uniform_real_distribution<float> distribution(0.0, 10.0);

  auto generate_float = [&generator, &distribution]() {
    return distribution(generator);
  };

  // Input to the SYCL device
  std::vector<float> A(vsize * vsize);
  std::vector<float> x(vsize);

  ranges::generate(A, generate_float);
  ranges::generate(x, generate_float);

  std::vector<float> output(vsize);

  {
    gstorm::sycl_exec exec;

    auto gpu_A = std::experimental::copy(exec, A);
    auto gpu_x = std::experimental::copy(exec, x);
    auto gpu_output = std::experimental::copy(exec, output);

    auto a_rows = gpu_A | ranges::view::chunk(vsize);
    auto x_repeat = ranges::view::repeat_n(gstorm::gpu::ref(gpu_x), vsize);

    auto zipped = ranges::view::zip(a_rows, x_repeat);

    std::experimental::transform(exec, zipped, gpu_output, DotProduct{});
  }

  auto expected = ranges::view::zip(A | ranges::view::chunk(vsize),
                    ranges::view::repeat_n(x, vsize))
                | ranges::view::transform(DotProduct{});

  EXPECT_TRUE(ranges::equal(expected, output));
}

TEST_F(Sgemv, sgemv) {
  const size_t vsize = 1024;

  std::default_random_engine generator;
  std::uniform_real_distribution<float> distribution(0.0, 10.0);

  auto generate_float = [&generator, &distribution]() {
    return distribution(generator);
  };

  // Input to the SYCL device
  std::vector<float> A(vsize * vsize);
  std::vector<float> x(vsize);
  std::vector<float> y(vsize);

  ranges::generate(A, generate_float);
  ranges::generate(x, generate_float);
  ranges::generate(y, generate_float);

  const auto a = generate_float();
  const auto b = generate_float();

  std::vector<float> output(vsize);

  {
    gstorm::sycl_exec exec;

    auto gpu_A = std::experimental::copy(exec, A);
    auto gpu_x = std::experimental::copy(exec, x);
    auto gpu_y = std::experimental::copy(exec, y);
    auto gpu_output = std::experimental::copy(exec, output);

    auto a_rows = gpu_A | ranges::view::chunk(vsize);
    auto x_repeat = ranges::view::repeat_n(gstorm::gpu::ref(gpu_x), vsize);

    auto dot_prod = ranges::view::zip(a_rows, x_repeat)
                  | ranges::view::transform(DotProduct{})
                  | ranges::view::transform([a](auto dot) { return dot * a; });

    auto zipped = my_zip(dot_prod, gpu_y | ranges::view::transform([b](auto y) {return b*y; }));

    std::experimental::transform(exec, zipped, gpu_output, MultiplyComponents{});
  }

  auto expected = ranges::view::zip(
                    ranges::view::zip(
                      A | ranges::view::chunk(vsize),
                      ranges::view::repeat_n(x, vsize))
                    | ranges::view::transform(DotProduct{})
                    | ranges::view::transform([a](auto dot) { return dot * a; }),
                    y | ranges::view::transform([b](auto y) {return b*y; }))
                | ranges::view::transform(MultiplyComponents{});

  EXPECT_TRUE(ranges::equal(expected, output));
}
