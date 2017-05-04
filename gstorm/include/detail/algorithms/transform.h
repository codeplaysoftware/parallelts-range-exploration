//
// Created by mhaidl on 23/08/16.
//

#pragma once

#include <CL/sycl.hpp>
#include <range/v3/all.hpp>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <meta/static_const.h>
#include <detail/algorithms/config.h>

namespace gstorm {
namespace gpu {
namespace algorithm {

class TransformKernel;

template<typename InRng, typename OutRng, typename UnaryFunc>
void transform(const InRng &in, OutRng &out, UnaryFunc func, cl::sycl::handler &cgh) {
  size_t distance = ranges::v3::distance(in);
  size_t thread_count = std::max(128ul, distance);
  cl::sycl::nd_range<1> config{distance, cl::sycl::range < 1 > {thread_count}};

  const auto inIt = in.begin();
  auto outIt = out.begin();

  cgh.parallel_for<class TransformKernel>(config, [=](cl::sycl::nd_item<1> id) {
    auto gid = static_cast<size_t>(id.get_global(0));
    *(outIt + gid) = func(*(inIt + gid));
  });
}

}
}
}
