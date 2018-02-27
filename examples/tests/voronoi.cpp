#include "gtest/gtest.h"

#include <fstream>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>

#include <gstorm.h>
#include <CL/sycl.hpp>
#include <range/v3/all.hpp>

#include "experimental.h"
#include "my_zip.h"

struct Voronoi : public testing::Test {};

struct minFunctor {
  int m, n, k;

  constexpr minFunctor(int m, int n, int k) : m(m), n(n), k(k) {}

  int minVoro(int x_i, int y_i, int p, int q) const {
    if (q == m * n) {
      return p;
    }

    // coordinates of points p and q
    int y_q = q / m;
    int x_q = q - y_q * m;
    int y_p = p / m;
    int x_p = p - y_p * m;

    // squared distances
    int d_iq = (x_i - x_q) * (x_i - x_q) + (y_i - y_q) * (y_i - y_q);
    int d_ip = (x_i - x_p) * (x_i - x_p) + (y_i - y_p) * (y_i - y_p);

    if (d_iq < d_ip) {
      return q;  // q is closer
    } else {
      return p;
    }
  }

  // For each point p+{-k,0,k}, we keep the Site with minimum distance
  template <typename T>
  int operator()(const T &t) const {
    // Current point and site
    int i = std::get<9>(t);
    int v = std::get<0>(t);

    // Current point coordinates
    int y = i / m;
    int x = i - y * m;

    if (x >= k) {
      v = minVoro(x, y, v, std::get<3>(t));

      if (y >= k) {
        v = minVoro(x, y, v, std::get<8>(t));
      }

      if (y + k < n) {
        v = minVoro(x, y, v, std::get<7>(t));
      }
    }

    if (x + m) {
      v = minVoro(x, y, v, std::get<1>(t));

      if (y >= k) {
        v = minVoro(x, y, v, std::get<6>(t));
      }

      if (y + k < n) {
        v = minVoro(x, y, v, std::get<5>(t));
      }
    }

    if (y >= k) {
      v = minVoro(x, y, v, std::get<4>(t));
    }

    if (y + k < n) {
      v = minVoro(x, y, v, std::get<2>(t));
    }

    // global return
    return v;
  }
};

template <typename T>
auto shift_range(T& range, int length, int amount) {
  auto cycled = ranges::view::cycle(range);
  return ranges::view::slice(cycled, length + amount, 2 * length + amount);
}

template <typename T>
auto shift_and_zip(T& range, int length, int k, int m, int n) {
  // Use transform with identity to force a load and strip the global address space
  auto i0 = range | ranges::view::transform([](auto a) { return a; });

  auto i1 = shift_range(i0, length, k);
  auto i2 = shift_range(i0, length, m * k);
  auto i3 = shift_range(i0, length, -k);
  auto i4 = shift_range(i0, length, - m * k);
  auto i5 = shift_range(i0, length, k + m * k);
  auto i6 = shift_range(i0, length, k - m * k);
  auto i7 = shift_range(i0, length, -k + m * k);
  auto i8 = shift_range(i0, length, -k - m * k);

  auto zr = ranges::view::zip(i0, i1, i2, i3, i4, i5, i6, i7, i8,
                              ranges::view::iota(0, length));

  return zr;
}

template <typename T>
void jfa(T& input, T& output, gstorm::sycl_exec& exec, int k, int m, int n) {
  auto length = m * n;
  auto zr = shift_and_zip(input, length, k, m, n);

  std::experimental::transform(exec, zr, output, minFunctor(m, n, k));
}

void generate_random_sites(std::vector<int> &t, int Nb, int m, int n) {
  std::default_random_engine rng;
  std::uniform_int_distribution<int> dist(0, m * n - 1);

  for (int k = 0; k < Nb; k++) {
    int index = dist(rng);
    t[index] = index + 1;
  }
}

TEST_F(Voronoi, TestVoronoi) {
  const int m = 512;  // number of rows
  const int n = 512;  // number of columns
  const int s = 100;  // number of sites
  const auto length = m * n;

  // Input to the SYCL device
  std::vector<int> hseeds(length, length);
  generate_random_sites(hseeds, s, m, n);

  std::vector<int> output(length, 0);
  {
    gstorm::sycl_exec exec;

    auto gpu_hseeds = std::experimental::copy(exec, hseeds);
    auto gpu_output = std::experimental::copy(exec, output);

    jfa(gpu_hseeds, gpu_output, exec, 1, m, n);
  }

  auto k = 1;
  auto& i = hseeds;

  auto expected = shift_and_zip(i, length, k, m, n)
                | ranges::view::transform(minFunctor(m, n, k));

  // Work around ranges::view::transform returning equal
  // iterators from begin & end when using cycle and slice
  for (auto i = 0; i < length; ++i) {
    EXPECT_EQ(expected[i], output[i]);
  }
}

// Export the tab to PGM image format
void vector_to_pgm(std::vector<int>& image, int m, int n, std::string filename) {
  std::ofstream outputFile(filename);
  outputFile << "P2\n";
  outputFile << m << " " << n << "\n255\n";

  for (auto value : image) {
    outputFile << (int) (71 * value) % 256; // Map values to [0,255]
    outputFile << " ";
  }

  outputFile << "\n";
}

TEST_F(Voronoi, TestVoronoiFull) {
  const int m = 512;  // number of rows
  const int n = 512;  // number of columns
  const int s = 100;  // number of sites

  // Input to the SYCL device
  std::vector<int> hseeds(m * n, m * n);
  generate_random_sites(hseeds, s, m, n);

  std::vector<int> output(m*n, m * n);
  {
    gstorm::sycl_exec exec;

    auto gpu_hseeds = std::experimental::copy(exec, hseeds);
    auto gpu_output = std::experimental::copy(exec, output);

    jfa(gpu_hseeds, gpu_output, exec, 1, m, n);
    std::swap(gpu_hseeds, gpu_output);

    // JFA : main loop with k=n/2, n/4, ..., 1
    for (int k = std::max(m, n) / 2; k > 0; k /= 2) {
      jfa(gpu_hseeds, gpu_output, exec, k, m, n);
      std::swap(gpu_hseeds, gpu_output);
    }
  }

  vector_to_pgm(hseeds, m, n, "discrete_voronoi.pgm");
}
