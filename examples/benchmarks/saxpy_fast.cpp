#include <chrono>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>

#include <gstorm.h>
#include <CL/sycl.hpp>
#include <range/v3/all.hpp>

#include <stdlib.h>

#include "experimental.h"
#include "my_zip.h"

template <typename T>
struct AddComponents {
  constexpr AddComponents(){};
  T operator()(const std::tuple<T, cl::sycl::global_ptr<T>>& tpl) const {
    return std::get<0>(tpl) + *std::get<1>(tpl);
  }
};

int main() {
  const size_t vsize = 1024 * 1024 * 16;
  const auto iterations = 100;

  std::default_random_engine generator;
  std::uniform_real_distribution<float> distribution(0.0, 10.0);

  auto generate_float = [&generator, &distribution]() {
    return distribution(generator);
  };

  auto x = static_cast<float*>(aligned_alloc(4096, vsize * sizeof(float)));
  auto y = static_cast<float*>(aligned_alloc(4096, vsize * sizeof(float)));
  auto z = static_cast<float*>(aligned_alloc(4096, vsize * sizeof(float)));

  for (auto i = 0u; i < vsize; ++i) {
    x[i] = generate_float();
    y[i] = generate_float();
  }
  // Input to the SYCL device
  // std::vector<float> x(vsize);
  // std::vector<float> y(vsize);

  // ranges::generate(x, generate_float);
  // ranges::generate(y, generate_float);

  const float a = generate_float();

  // std::vector<float> z(vsize);
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

      // auto gpu_x = std::experimental::copy(exec, x);
      // auto gpu_y = std::experimental::copy(exec, y);
      // auto gpu_z = std::experimental::copy(exec, z);
      auto gpu_x = gstorm::range::gvector<std::vector<float>>(x, vsize);
      auto gpu_y = gstorm::range::gvector<std::vector<float>>(y, vsize);
      auto gpu_z = gstorm::range::gvector<std::vector<float>>(z, vsize);

      exec.registerGVector(&gpu_x);
      exec.registerGVector(&gpu_y);
      exec.registerGVector(&gpu_z);

      auto multiply_components = [](const auto& tpl) {
        return std::get<0>(tpl) * *std::get<1>(tpl);
      };

      // auto ax = my_zip(ranges::view::repeat(a), gpu_x)
      //         | ranges::view::transform(multiply_components);

      auto mult_with_a = [a](auto tpl) { return tpl * a; };
      auto ax = gpu_x | ranges::view::transform(mult_with_a);

      std::experimental::transform(exec, my_zip(ax, gpu_y), gpu_z,
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

  free(x);
  free(y);
  free(z);
}
