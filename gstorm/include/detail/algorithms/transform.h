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

template<typename InRng1, typename InRng2, typename OutRng, typename BinaryFunc>
void transform(const InRng1 &in1, const InRng2& in2, OutRng &out, BinaryFunc func, cl::sycl::handler &cgh) {
  const size_t global_thread_count = ranges::distance(in1);
  const size_t local_thread_count = ranges::min(128ul, global_thread_count);

  cl::sycl::nd_range<1> config{global_thread_count, local_thread_count};

  const auto inIt1 = in1.begin();
  const auto inIt2 = in2.begin();
  auto outIt = out.begin();

  cgh.parallel_for<class TransformKernel<1, BinaryFunc>>(config, [=](cl::sycl::nd_item<1> id) {
    auto gid = static_cast<size_t>(id.get_global(0));
    *(outIt + gid) = func(*(inIt1 + gid), *(inIt2 + gid));
  });
}

}
}
}
