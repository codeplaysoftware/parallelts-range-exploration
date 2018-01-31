//
// Created by mhaidl on 23/08/16.
//

#pragma once

#include <cstddef>
#include <type_traits>
#include <utility>

#include <CL/sycl.hpp>
#include <range/v3/all.hpp>

namespace gstorm {
namespace gpu {
namespace algorithm {

template <int x, typename... Ts>
class TransformKernel {};

template <typename InRng, typename OutRng, typename UnaryFunc>
void transform(const InRng &in, OutRng &out, UnaryFunc func,
               cl::sycl::handler &cgh) {
  const auto default_local_size = 128ul;
  const auto range_length = static_cast<std::size_t>(ranges::distance(in));

  const auto global_thread_count =
      range_length % default_local_size == 0
          ? range_length
          : ((range_length / default_local_size) + 1) * default_local_size;

  const auto local_thread_count =
      ranges::min(default_local_size, global_thread_count);

  cl::sycl::nd_range<1> config{global_thread_count, local_thread_count};

  const auto inIt = in.begin();
  auto outIt = out.begin();

  cgh.parallel_for<class TransformKernel<1, UnaryFunc>>(config,
      [range_length, func, inIt, outIt](cl::sycl::nd_item<1> id) {
    auto gid = id.get_global(0);
    if (gid < range_length) {
      *(outIt + gid) = func(*(inIt + gid));
    }
  });
}
}
}
}
