#include "gtest/gtest.h"

#include <gstorm.h>
#include <fstream>
#include <vector>

#include <CL/sycl.hpp>
#include <range/v3/all.hpp>

#include "experimental.h"
#include "my_zip.h"

struct Mandelbrot : public testing::Test {};

/**
 * Store an RGB pixel with 8-bits per colour.
 */
struct pixel {
  unsigned char r, g, b, a;
  //  unsigned int r : 8; // changed to a 32 bit field
  //  unsigned int g : 8; // this results in a single 32 bit store
  //  unsigned int b : 8; // not 4 8 bit stores for each value
  //  unsigned int a : 8;

  /**
   * Create a pixel and assign it some colour based on ``n``.
   *
   * @param n A number to map to the RGB colour space
   */
  pixel(int n) {
    r = (n & 63) << 2;
    g = (n << 3) & 255;
    b = (n >> 8) & 255;
    a = 255;
  }

  /**
   * Create a pixel using the specified RGB values.
   *
   * @param r Red value
   * @param g Green value
   * @param b Blue value
   */
  pixel(unsigned char r = 0, unsigned char g = 0, unsigned char b = 0)
      : r(r), g(g), b(b) {}
};

bool operator==(const pixel& lhs, const pixel& rhs) {
  return lhs.r == rhs.r && lhs.g == rhs.g && lhs.b == rhs.b && lhs.a == rhs.a;
}

std::ostream& operator<<(std::ostream& out, pixel p) {
  out << (int)p.r << " " << (int)p.g << " " << (int)p.b << "\n";
  return out;
}

struct CalculatePixel {
  const int height;
  const int width;
  const int niters;

  constexpr CalculatePixel(int height, int width, int niters)
      : height{height}, width{width}, niters{niters} {};

  auto operator()(const int& index) const {
    auto x = index / width;
    auto y = x ? index % (x * width) : index;
    auto Zr = 0.0f;
    auto Zi = 0.0f;

    const auto radius = 2.0f;
    const auto radius_squared = radius * radius;

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

template <typename Rng>
void vector_to_ppm(Rng& rng, unsigned width, unsigned height,
                   const std::string& filename) {
  std::ofstream outputFile(filename);

  outputFile << "P3\n" << width << " " << height << "\n255\n";
  for (const auto& v : rng)
    outputFile << v;
}

TEST_F(Mandelbrot, TestMandelbrot) {
  const auto height = 512;
  const auto width = 512;
  const auto iterations = 100;

  std::vector<pixel> image(height * width);

  {
    gstorm::sycl_exec exec;

    auto gpu_image = std::experimental::copy(exec, image);

    auto indices = ranges::view::iota(0) | ranges::view::take(width * height);
    std::experimental::transform(exec, indices, gpu_image,
                                 CalculatePixel{height, width, iterations});
  }

  auto expected = ranges::view::iota(0)
                | ranges::view::take(width * height)
                | ranges::view::transform(CalculatePixel{height, width, iterations});

  EXPECT_TRUE(ranges::equal(
      expected, image,
      [](const auto& lhs, const auto& rhs) { return lhs == rhs; }));

  vector_to_ppm(image, width, height, "mandelbrot.ppm");
}
