#include <chrono>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

#include <gstorm.h>
#include <CL/sycl.hpp>
#include <range/v3/all.hpp>

#include "aligned_allocator.h"
#include "experimental.h"
#include "my_zip.h"

struct pixel {
  unsigned char r, g, b, a;
  //  unsigned int r : 8; // changed to a 32 bit field
  //  unsigned int g : 8; // this results in a single 32 bit store
  //  unsigned int b : 8; // not 4 8 bit stores for each value
  //  unsigned int a : 8;

  pixel(int n) {
    r = (n & 63) << 2;
    g = (n << 3) & 255;
    b = (n >> 8) & 255;
    a = 255;
  }
  pixel(unsigned char r = 0, unsigned char g = 0, unsigned char b = 0)
      : r(r), g(g), b(b) {}
};

bool operator==(const pixel& lhs, const pixel& rhs) {
  return lhs.r == rhs.r && lhs.g == rhs.g && lhs.b == rhs.b && lhs.a == rhs.a;
}

struct CalculatePixel {
  const int height;
  const int width;
  const int niters;

  constexpr CalculatePixel(int height, int width, int niters)
      : height{height}, width{width}, niters{niters} {};

  auto operator()(const int& index) const {
    const auto x = index / width;
    const auto y = x ? index % (x * width) : index;
    auto Zr = 0.0f;
    auto Zi = 0.0f;

    const auto radius = 2.0f;
    const auto radius_squared = radius*radius;

    auto Cr = (y * (radius / height) - 1.5f);
    auto Ci = (x * (radius / width) - 1.0f);

    int value = 0;
    for (value = 0; value < niters; ++value) {
      const auto ZiN = Zi * Zi;
      const auto ZrN = Zr * Zr;
      if (ZiN + ZrN > radius_squared) {
        break;
      }
      Zi *= Zr;
      Zi *= radius;
      Zi += Ci;
      Zr = ZrN - ZiN + Cr;
    }
    return pixel(value);
  }
};

int main(int argc, char* argv[]) {
  const std::size_t base_size = 128;

  std::size_t multiplier = 1;

  if (argc > 1) {
    std::stringstream{argv[1]} >> multiplier;
  }

  std::cout << "Size: " << multiplier << "\n";
  const auto vsize = static_cast<int>(1024 * multiplier);
  const auto iterations = 100;
  std::vector<pixel, aligned_allocator<pixel, 4096>> image(vsize * base_size);

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

      auto gpu_image = std::experimental::copy(exec, image);

      auto indices = ranges::view::iota(0)
                   | ranges::view::take(vsize * base_size);
      std::experimental::transform(
          exec, indices, gpu_image,
          CalculatePixel{vsize, base_size, iterations});
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

  auto expected = ranges::view::iota(0)
                | ranges::view::take(vsize * base_size)
                | ranges::view::transform(CalculatePixel{vsize, base_size, iterations});

  if (!ranges::equal(expected, image, [](const auto& lhs, const auto& rhs) {
        return lhs == rhs;
      })) {
    std::cout << "Mismatch between expected and actual result!\n";
    return 1;
  }
}
