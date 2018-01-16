#include <chrono>
#include <iostream>
#include <random>
#include <sstream>
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
  T operator()(const std::tuple<cl::sycl::global_ptr<T>,
                                cl::sycl::global_ptr<T>>& tpl) const {
    return *std::get<0>(tpl) + *std::get<1>(tpl);
  }
};

template <typename T>
struct MultiplyComponents {
  constexpr MultiplyComponents(){};
  T operator()(const std::tuple<cl::sycl::global_ptr<T>,
                                cl::sycl::global_ptr<T>>& tpl) const {
    return *std::get<0>(tpl) * *std::get<1>(tpl);
  }
};

int main(int argc, char* argv[]) {
  const size_t base_size = 1024 * 1024;

  size_t multiplier = 16;
  if (argc > 1) {
    std::stringstream{argv[1]} >> multiplier;
  }

  std::cout << "Size: " << multiplier << "\n";
  const auto vsize = base_size * multiplier;
  const auto iterations = 100;

  std::default_random_engine generator;
  std::uniform_real_distribution<float> distribution(-1.0, 1.0);

  auto generate_float = [&generator, &distribution]() {
    return distribution(generator);
  };
  auto x = static_cast<float*>(aligned_alloc(4096, vsize * sizeof(float)));
  auto y = static_cast<float*>(aligned_alloc(4096, vsize * sizeof(float)));
  auto tmp = static_cast<float*>(aligned_alloc(4096, vsize * sizeof(float)));

  for (auto i = 0u; i < vsize; ++i) {
    x[i] = generate_float();
    y[i] = generate_float();
  }

  // Input to the SYCL device
  // std::vector<float> x(vsize);
  // std::vector<float> y(vsize);

  // ranges::generate(x, generate_float);
  // ranges::generate(y, generate_float);

  // std::vector<float> tmp(vsize);

  auto result = 0.0f;
  std::vector<double> times{};

  cl::sycl::gpu_selector device_selector;
  auto q = cl::sycl::queue(device_selector);
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
      // auto gpu_tmp = std::experimental::copy(exec, tmp);

      auto gpu_x = gstorm::range::gvector<std::vector<float>>(x, vsize);
      auto gpu_y = gstorm::range::gvector<std::vector<float>>(y, vsize);
      auto gpu_tmp = gstorm::range::gvector<std::vector<float>>(tmp, vsize);

      exec.registerGVector(&gpu_x);
      exec.registerGVector(&gpu_y);
      exec.registerGVector(&gpu_tmp);

      auto zipped = my_zip(gpu_x, gpu_y);
      std::experimental::transform(exec, zipped, gpu_tmp,
                                   MultiplyComponents<float>{});

      result =
          std::experimental::reduce(exec, gpu_tmp, 0.0f, std::plus<float>{});
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

  auto expected = 0.0f;
  for (auto i = 0u; i < vsize; ++i) {
    expected += x[i] * y[i];
  }

  // Is within 0.1%?
  if (std::abs(result - expected) >
      0.001 * std::max(std::abs(result), std::abs(expected))) {
    std::cout << "Mismatch between expected and actual result!\n";
  }

  free(x);
  free(y);
  free(tmp);
}
