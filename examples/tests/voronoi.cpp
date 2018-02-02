#include "gtest/gtest.h"

#include <gstorm.h>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>
#include <iostream>

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

  // Input to the SYCL device
  std::vector<int> hseeds(m * n, m * n);
  generate_random_sites(hseeds, s, m, n);

  std::vector<int> output(261631, m * n);
  {
    gstorm::sycl_exec exec;

    auto gpu_hseeds = std::experimental::copy(exec, hseeds);
    auto gpu_output = std::experimental::copy(exec, output);

    // Use transform with identity to force a load and strip the global address space
    auto i = gpu_hseeds | ranges::view::transform([](auto a) { return a; });
    auto k = 1;

    auto i0 = ranges::view::unbounded(i.begin());
    auto i1 = ranges::view::unbounded(i.begin() - k);
    auto i2 = ranges::view::unbounded(i.begin() - m * k);
    auto i3 = ranges::view::unbounded(i.begin() + k - m * k);
    auto i4 = ranges::view::unbounded(i.begin() - k + m * k);
    auto i5 = ranges::view::unbounded(i.begin() - k - m * k);

    auto zr = ranges::view::zip(i0, ranges::view::slice(i, k, n * m),
                                ranges::view::slice(i, m * k, n * m), i1, i2,
                                ranges::view::slice(i, k + m * k, n * m), i3,
                                i4, i5, ranges::view::iota(0, n * m));

    auto zrt = zr | ranges::view::take(n * m);
    std::experimental::transform(exec, zrt, gpu_output, minFunctor(m, n, k));
  }

  auto k = 1;
  auto& i = hseeds;

  auto i0 = ranges::view::unbounded(i.begin());
  auto i1 = ranges::view::unbounded(i.begin() - k);
  auto i2 = ranges::view::unbounded(i.begin() - m * k);
  auto i3 = ranges::view::unbounded(i.begin() + k - m * k);
  auto i4 = ranges::view::unbounded(i.begin() - k + m * k);
  auto i5 = ranges::view::unbounded(i.begin() - k - m * k);

  auto zr = ranges::view::zip(i0, ranges::view::slice(i, k, n * m),
                              ranges::view::slice(i, m * k, n * m), i1, i2,
                              ranges::view::slice(i, k + m * k, n * m), i3,
                              i4, i5, ranges::view::iota(0, n * m));

  auto expected = zr
                | ranges::view::take(n * m)
                | ranges::view::transform(minFunctor(m, n, k));

  EXPECT_TRUE(ranges::equal(expected, output));
}
