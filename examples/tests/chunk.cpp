#include "gtest/gtest.h"

#include <gstorm.h>
#include <vector>
#include <random>

#include <range/v3/all.hpp>
#include <CL/sycl.hpp>

#include "experimental.h"

struct Chunk : public testing::Test {};

struct Sum {
  constexpr Sum() {};
  template<typename T>
  float operator()(const T& row) const {
    return ranges::accumulate(row, 0.0f);
  }
};

TEST_F(Chunk, SumRows) {

  const size_t vsize = 1024;

  std::default_random_engine generator;
  std::uniform_real_distribution<float> distribution(0.0,10.0);

  auto generate_float =
    [&generator, &distribution]() { return distribution(generator); };

  // Input to the SYCL device
  std::vector<float> A(vsize*vsize);

  ranges::generate(A, generate_float);

  std::vector<float> output(vsize);

  {
    gstorm::sycl_exec exec;

    auto gpu_A = std::experimental::copy(exec, A);
    auto gpu_output = std::experimental::copy(exec, output);

    auto a_row = gpu_A | ranges::view::chunk(vsize);

    std::experimental::transform(exec, a_row, gpu_output, Sum{});
  }

  auto expected = A
                | ranges::view::chunk(vsize)
                | ranges::view::transform(Sum{});

  EXPECT_TRUE(ranges::equal(expected, output));
}
