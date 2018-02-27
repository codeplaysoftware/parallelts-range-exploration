#include "gtest/gtest.h"

#include <gstorm.h>
#include <random>
#include <tuple>
#include <vector>

#include <CL/sycl.hpp>
#include <range/v3/all.hpp>

#include "experimental.h"
#include "my_zip.h"

struct NBody : public testing::Test {};

struct NBodyOperator {
  float G, dt, eps2;

  constexpr NBodyOperator(float G, float dt, float eps2)
      : G{G}, dt{dt}, eps2{eps2} {};

  template <typename T>
  auto operator()(const T& tpl) const {
    return calculate(*std::get<0>(tpl), *std::get<1>(tpl), std::get<2>(tpl));
  }

  template <typename T>
  cl::sycl::float4 calculate(const cl::sycl::float4& p_orig,
                             cl::sycl::float4& v_orig, T& particles) const {
    auto p = p_orig;
    auto v = v_orig;
    cl::sycl::float4 a{0.0f, 0.0f, 0.0f, 0.0f};
    cl::sycl::float4 r{0.0f, 0.0f, 0.0f, 0.0f};

    for (auto& particle : particles) {
      r.x() = p.x() - particle.x();
      r.y() = p.y() - particle.y();
      r.z() = p.z() - particle.z();
      r.w() =
          cl::sycl::sqrt(r.x() * r.x() + r.y() * r.y() + r.z() * r.z() + eps2);

      a.w() = G * particle.w() * r.w() * r.w() * r.w();

      a.x() += a.w() * r.x();
      a.y() += a.w() * r.y();
      a.z() += a.w() * r.z();
    }

    p.x() += v.x() * dt + a.x() * 0.5f * dt * dt;
    p.y() += v.y() * dt + a.y() * 0.5f * dt * dt;
    p.z() += v.z() * dt + a.z() * 0.5f * dt * dt;

    v.x() += a.x() * dt;
    v.y() += a.y() * dt;
    v.z() += a.z() * dt;

    v_orig = v;
    return p;
  }
};

bool float_comparison(float result, float expected) {
  return std::abs(result - expected) <=
         0.001 * std::max(std::abs(result), std::abs(expected));
}

bool float4_comparison(cl::sycl::float4 left, cl::sycl::float4 right) {
  return float_comparison(left.x(), right.x()) &&
         float_comparison(left.y(), right.y()) &&
         float_comparison(left.z(), right.z()) &&
         float_comparison(left.w(), right.w());
}

TEST_F(NBody, TestNBody) {
  const size_t vsize = 512;

  std::default_random_engine generator;
  std::uniform_real_distribution<float> distribution(-5.0, 5.0);

  auto generate_float = [&generator, &distribution]() {
    return distribution(generator);
  };

  auto generate_float4 = [&generate_float]() {
    return cl::sycl::float4{generate_float(), generate_float(),
                            generate_float(), generate_float()};
  };

  const auto G = -6.673e-11f;
  const auto dt = 3600.f;
  const auto eps2 = 0.00125f;

  // Input to the SYCL device
  std::vector<cl::sycl::float4> position(vsize);
  std::vector<cl::sycl::float4> velocity(vsize);
  std::vector<cl::sycl::float4> new_position(vsize);

  ranges::generate(position, generate_float4);
  ranges::generate(velocity, generate_float4);

  auto expected_velocity = velocity;
  {
    gstorm::sycl_exec exec;

    auto gpu_position = std::experimental::copy(exec, position);
    auto gpu_velocity = std::experimental::copy(exec, velocity);
    auto gpu_new_position = std::experimental::copy(exec, new_position);

    auto repeated =
        ranges::view::repeat_n(gstorm::gpu::ref(gpu_position), vsize);

    auto zipped = my_zip(gpu_position, gpu_velocity, repeated);

    std::experimental::transform(exec, zipped, gpu_new_position,
                                 NBodyOperator{G, dt, eps2});
  }

  auto op = [G, dt, eps2](const auto& tpl) -> cl::sycl::float4 {
    return (NBodyOperator{G, dt, eps2})
        .calculate(std::get<0>(tpl), std::get<1>(tpl), std::get<2>(tpl));
  };

  auto expected_position = ranges::view::zip(position, expected_velocity,
                             ranges::view::repeat_n(position, vsize))
                         | ranges::view::transform(op);

  EXPECT_TRUE(
      ranges::equal(expected_position, new_position, float4_comparison));
  EXPECT_TRUE(ranges::equal(expected_velocity, velocity, float4_comparison));
}
