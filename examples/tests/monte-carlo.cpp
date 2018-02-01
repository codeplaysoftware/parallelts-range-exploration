#include "gtest/gtest.h"

#include <gstorm.h>
#include <vector>
#include <cmath>
#include <functional>
#include <range/v3/all.hpp>

#include "experimental.h"

struct MonteCarlo : public testing::Test {};

// Redirect undefined functions on the device to sycl equivalents
#ifdef __SYCL_DEVICE_ONLY__
extern "C" {
long double logl(long double f) {
  return cl::sycl::log(cl_double(f));
}

float nextafterf(float f, float g) {
  return cl::sycl::nextafter(f, g);
}
}
#endif

struct estimate_pi {
  float operator()(unsigned int thread_id) const {
    float sum = 0;
    unsigned int N = 5000; // samples per stream

    // create a random number generator
    std::default_random_engine rng(thread_id);

    // create a mapping from random numbers to [0,1)
    // needs the redirections to sycl equivalents defined above
    std::uniform_real_distribution<float> u01(0, 1);

    // Workaround to be able to use device only extern "C" functions
    // needed by uniform_real_distribution. Without explicitly calling
    // them here, they still end up undefined.
    nextafterf(1.f,2.f);
    logl(1.f);

    // take N samples in a quarter circle
    for (unsigned int i = 0; i < N; ++i) {
      // draw a sample from the unit square
      float x = u01(rng);
      float y = u01(rng);

      // measure distance from the origin
      float dist = cl::sycl::sqrt(x * x + y * y);

      // add 1.0f if (u0,u1) is inside the quarter circle
      if (dist <= 1.0f)
        sum += 1.0f;
    }

    // multiply by 4 to get the area of the whole circle
    sum *= 4.0f;

    // divide by N
    return sum / N;
  }
};

TEST_F(MonteCarlo, TestMonteCarlo) {

  const auto PI = 3.141592653589793;

  const size_t M = 1024;

  {
    gstorm::sycl_exec exec;

    auto estimations = ranges::view::iota(0)
                     | ranges::view::take(M)
                     | ranges::view::transform(estimate_pi{});

    auto result = std::experimental::reduce(exec, estimations, 0.0f, std::plus<float>{});

    result /= M;

    // Is within 1%?
    EXPECT_LT(std::abs(result - PI), 0.01*PI);
  }
}
