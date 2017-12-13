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

template <int x, typename... Ts>
class TransformKernel{};

template<typename InRng, typename OutRng, typename UnaryFunc>
void transform(const InRng &in, OutRng &out, UnaryFunc func, cl::sycl::handler &cgh) {
  const size_t size = ranges::v3::distance(in);
  const size_t global_thread_count = size % 128ul == 0ul ? size: (size/128ul + 1) * 128ul;
  const size_t local_thread_count = ranges::min(128ul, global_thread_count);

  cl::sycl::nd_range<1> config{global_thread_count, local_thread_count};

  const auto inIt = in.begin();
  auto outIt = out.begin();

  cgh.parallel_for<class TransformKernel<1, UnaryFunc>>(config, [=](cl::sycl::nd_item<1> id) {
    auto gid = static_cast<size_t>(id.get_global(0));
    if (gid < size)
      *(outIt + gid) = func(*(inIt + gid));
  });
}

}
}
}
