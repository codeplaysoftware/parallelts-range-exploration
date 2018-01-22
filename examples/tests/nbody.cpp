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
    data_t a = {0.0f, 0.0f, 0.0f, 0.0f};
    data_t r = {0.0f, 0.0f, 0.0f, 0.0f};

    auto p = *std::get<0>(tpl);
    auto v = *std::get<1>(tpl);
    auto& particles = std::get<2>(tpl);

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

    *std::get<1>(tpl) = v;
    return p;
  }
};

TEST_F(NBody, TestNBody) {
  const size_t vsize = 1024;

  std::default_random_engine generator;
  std::uniform_real_distribution<float> distribution(-10.0, 10.0);

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
  std::vector<data_t> y(vsize);

  ranges::generate(position, generate_data_t);
  ranges::generate(position, generate_data_t);

  {
    gstorm::sycl_exec exec;

    auto gpu_position = std::experimental::copy(exec, position);
    auto gpu_velocity = std::experimental::copy(exec, velocity);
    auto gpu_y = std::experimental::copy(exec, y);

    auto repeated =
        ranges::view::repeat_n(gstorm::gpu::ref(gpu_position), vsize);

    auto zipped = my_zip(gpu_position, gpu_velocity, repeated);

    std::experimental::transform(exec, zipped, gpu_y,
                                 NBodyOperator{G, dt, eps2});
  }

  // EXPECT_TRUE(ranges::equal(expected, z));
}
