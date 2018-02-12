#include <chrono>
#include <iostream>
#include <random>
#include <sstream>
#include <tuple>
#include <vector>

#include <gstorm.h>
#include <CL/sycl.hpp>
#include <range/v3/all.hpp>

#include "aligned_allocator.h"
#include "experimental.h"
#include "my_zip.h"

template <typename T>
struct AddComponents {
  constexpr AddComponents(){};
  T operator()(const std::tuple<cl::sycl::global_ptr<T>,
                                cl::sycl::global_ptr<T>>& tpl) const {
    return *std::get<0>(tpl) + *std::get<1>(tpl);
  }
};

template <typename T>
struct MultiplyComponents {
  constexpr MultiplyComponents(){};
  T operator()(const std::tuple<T, cl::sycl::global_ptr<T>>& tpl) const {
    return std::get<0>(tpl) * *std::get<1>(tpl);
  }
};

template <typename T>
struct MultiplyWithA {
  T a;
  constexpr MultiplyWithA(T a) : a{a} {};
  T operator()(T x) const { return x * a; }
};

int main(int argc, char* argv[]) {
  const size_t base_size = 1024 * 128;

  size_t multiplier = 16;
  if (argc > 1) {
    std::stringstream{argv[1]} >> multiplier;
  }

  std::cout << "Size: " << multiplier << "\n";
  const auto vsize = base_size * multiplier;
  const auto iterations = 100;

  std::default_random_engine generator;
  std::uniform_real_distribution<float> distribution(0.0, 10.0);

  auto generate_float = [&generator, &distribution]() {
    return distribution(generator);
  };

  // Input to the SYCL device
  std::vector<float, aligned_allocator<float, 4096>> x(vsize);
  std::vector<float, aligned_allocator<float, 4096>> y(vsize);

  ranges::generate(x, generate_float);
  ranges::generate(y, generate_float);

  const float a = generate_float();

  std::vector<float, aligned_allocator<float, 4096>> ax(vsize);
  std::vector<float, aligned_allocator<float, 4096>> z(vsize);

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

      auto gpu_x = std::experimental::copy(exec, x);
      auto gpu_y = std::experimental::copy(exec, y);
      auto gpu_ax = std::experimental::copy(exec, ax);
      auto gpu_z = std::experimental::copy(exec, z);

      std::experimental::transform(
          exec, my_zip(ranges::view::repeat_n(a, vsize), gpu_x), gpu_ax,
          MultiplyComponents<float>{});

      std::experimental::transform(exec, my_zip(gpu_ax, gpu_y), gpu_z,
                                   AddComponents<float>{});
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

  auto expected =
    ranges::view::transform(x, y, [a](auto x, auto y) { return x*a + y; });

  if (!ranges::equal(expected, z)) {
    std::cout << "Mismatch between expected and actual result!\n";
    return 1;
  }
}
