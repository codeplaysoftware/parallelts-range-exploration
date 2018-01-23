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

using data_t = struct F4 {
  F4() : x(0), y(0), z(0), w(0) {}
  F4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
  float x, y, z, w;
};

struct NBodyOperator {
  float G, dt, eps2;

  constexpr NBodyOperator(float G, float dt, float eps2)
      : G{G}, dt{dt}, eps2{eps2} {};

  template <typename T>
  auto operator()(const T& tpl) const {
    return calculate(*std::get<0>(tpl), *std::get<1>(tpl), std::get<2>(tpl));
  }

  template <typename T>
  data_t calculate(const data_t& p_orig, data_t& v_orig, T& particles) const {
    auto p = p_orig;
    auto v = v_orig;
    data_t a = {0.0f, 0.0f, 0.0f, 0.0f};
    data_t r = {0.0f, 0.0f, 0.0f, 0.0f};

    for (auto& particle : particles) {
      r.x = p.x - particle.x;
      r.y = p.y - particle.y;
      r.z = p.z - particle.z;
      r.w = cl::sycl::sqrt(r.x * r.x + r.y * r.y + r.z * r.z + eps2);

      a.w = G * particle.w * r.w * r.w * r.w;

      a.x += a.w * r.x;
      a.y += a.w * r.y;
      a.z += a.w * r.z;
    }

    p.x += v.x * dt + a.x * 0.5f * dt * dt;
    p.y += v.y * dt + a.y * 0.5f * dt * dt;
    p.z += v.z * dt + a.z * 0.5f * dt * dt;

    v.x += a.x * dt;
    v.y += a.y * dt;
    v.z += a.z * dt;

    v_orig = v;
    return p;
  }
};

bool float_comparison(float result, float expected) {
  return std::abs(result - expected) <=
         0.001 * std::max(std::abs(result), std::abs(expected));
}

bool data_t_comparison(data_t left, data_t right) {
  return float_comparison(left.x, right.x) &&
         float_comparison(left.y, right.y) &&
         float_comparison(left.z, right.z) && float_comparison(left.w, right.w);
}

TEST_F(NBody, TestNBody) {
  const size_t vsize = 512;

  std::default_random_engine generator;
  std::uniform_real_distribution<float> distribution(-5.0, 5.0);

  auto generate_float = [&generator, &distribution]() {
    return distribution(generator);
  };

  auto generate_data_t = [&generate_float]() {
    return data_t{generate_float(), generate_float(), generate_float(),
                  generate_float()};
  };

  const auto G = -6.673e-11f;
  const auto dt = 3600.f;
  const auto eps2 = 0.00125f;

  // Input to the SYCL device
  std::vector<data_t> position(vsize);
  std::vector<data_t> velocity(vsize);
  std::vector<data_t> new_position(vsize);

  ranges::generate(position, generate_data_t);
  ranges::generate(velocity, generate_data_t);

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

  auto op = [G, dt, eps2](const auto& tpl) -> data_t {
    return (NBodyOperator{G, dt, eps2})
        .calculate(std::get<0>(tpl), std::get<1>(tpl), std::get<2>(tpl));
  };

  auto expected_position = ranges::view::zip(position, expected_velocity,
                             ranges::view::repeat_n(position, vsize))
                         | ranges::view::transform(op);

  EXPECT_TRUE(
      ranges::equal(expected_position, new_position, data_t_comparison));
  EXPECT_TRUE(ranges::equal(expected_velocity, velocity, data_t_comparison));
}
