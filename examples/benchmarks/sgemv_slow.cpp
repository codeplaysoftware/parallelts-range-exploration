#include <chrono>
#include <iostream>
#include <random>
#include <sstream>
#include <utility>
#include <vector>

#include <gstorm.h>
#include <CL/sycl.hpp>
#include <range/v3/all.hpp>

#include "aligned_allocator.h"
#include "experimental.h"
#include "my_zip.h"

struct DotProduct {
  constexpr DotProduct(){};
  template <typename T>
  float operator()(const T& tpl) const {
    auto row = std::get<0>(tpl);
    auto x = std::get<1>(tpl);

    return ranges::inner_product(row, x, 0.0f);
  }
};

struct AddComponents {
  constexpr AddComponents() {};

  template <typename T>
  int operator()(const T& a) const {
    return *std::get<0>(a) + *std::get<1>(a);
  }
};

struct MultiplyWith {
  float x;
  constexpr MultiplyWith(float x) : x{x} {};

  float operator()(float a) const {
    return a * x;
  }
};

int main(int argc, char* argv[]) {
  const size_t base_size = 128;

  size_t multiplier = 1;

  if (argc > 1) {
    std::stringstream{argv[1]} >> multiplier;
  }

  std::cout << "Size: " << multiplier << "\n";
  const auto vsize = 1024 * multiplier;
  const auto iterations = 100;

  std::default_random_engine generator;
  std::uniform_real_distribution<float> distribution(0.0, 10.0);

  auto generate_float = [&generator, &distribution]() {
    return distribution(generator);
  };

  // Input to the SYCL device
  std::vector<float, aligned_allocator<float, 4096>> A(vsize * base_size);
  std::vector<float, aligned_allocator<float, 4096>> x(base_size);
  std::vector<float, aligned_allocator<float, 4096>> y(vsize);


  ranges::generate(A, generate_float);
  ranges::generate(x, generate_float);
  ranges::generate(y, generate_float);

  const auto a = generate_float();
  const auto b = generate_float();

  std::vector<float, aligned_allocator<float, 4096>> output(vsize);

  std::vector<float, aligned_allocator<float, 4096>> tmp_dot(vsize);
  std::vector<float, aligned_allocator<float, 4096>> tmp_scale_x(vsize);
  std::vector<float, aligned_allocator<float, 4096>> tmp_scale_y(vsize);


  std::vector<double> times{};

  cl::sycl::gpu_selector device_selector;
  auto q = cl::sycl::queue(device_selector,
                           {cl::sycl::property::queue::enable_profiling{}});
  std::cout << "Using device: "
            << q.get_device().get_info<cl::sycl::info::device::name>()
            << ", from: "
            << q.get_device()
                   .get_platform()
                   .get_info<cl::sycl::info::platform::name>()
            << "\n";

  for (auto i = 0; i < iterations; ++i) {
    auto start = std::chrono::system_clock::now();
    {
      gstorm::sycl_exec exec(q);

      auto gpu_A = std::experimental::copy(exec, A);
      auto gpu_x = std::experimental::copy(exec, x);
      auto gpu_y = std::experimental::copy(exec, y);
      auto gpu_output = std::experimental::copy(exec, output);

      auto gpu_dot = std::experimental::copy(exec, tmp_dot);
      auto gpu_scale_x = std::experimental::copy(exec, tmp_scale_x);
      auto gpu_scale_y = std::experimental::copy(exec, tmp_scale_y);

      auto a_rows = gpu_A | ranges::view::chunk(base_size);
      auto x_repeat = ranges::view::repeat_n(gstorm::gpu::ref(gpu_x), vsize);

      auto zip1 =
          ranges::view::zip(a_rows, x_repeat);

      std::experimental::transform(exec, zip1, gpu_dot, DotProduct{});

      std::experimental::transform(exec, gpu_scale_x, gpu_dot, MultiplyWith{a});

      std::experimental::transform(exec, gpu_scale_y, gpu_y, MultiplyWith{b});

      auto zipped = my_zip(gpu_scale_x, gpu_scale_y);

      std::experimental::transform(exec, zipped, gpu_output,
                                   AddComponents{});
    }

    auto end = std::chrono::system_clock::now();

    auto time_taken =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    times.push_back(time_taken.count() / 1000.0);
    std::cout << "\r" << (i + 1) << "/" << iterations << std::flush;
  }
  std::cout << "\n";

  ranges::sort(times);
  std::cout << "Median time: " << times[iterations / 2] << " ms\n";

  auto expected = ranges::view::zip(
                    ranges::view::zip(
                      A | ranges::view::chunk(base_size),
                      ranges::view::repeat_n(x, vsize))
                    | ranges::view::transform(DotProduct{})
                    | ranges::view::transform([a](auto dot) { return dot * a; }),
                    y | ranges::view::transform([b](auto y) {return b*y; }))
                | ranges::view::transform([](const auto& tpl)
                    {return std::get<0>(tpl) + std::get<1>(tpl);} );

  if (!ranges::equal(expected, output)) {
    std::cout << "Mismatch between expected and actual result!\n";
    return 1;
  }
}
