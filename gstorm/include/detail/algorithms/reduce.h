//
// Created by mhaidl on 23/08/16.
//

#pragma once

#include <range/v3/all.hpp>

namespace gstorm {
namespace gpu {
namespace algorithm {

template <int x, typename... Ts>
class ReduceKernel{};

template<typename InRng, typename T, typename BinaryFunc, typename value_type = T>// std::remove_cv_t<std::remove_reference_t<decltype(*(InRng::begin()))>>>
auto reduce(InRng &in, T init, BinaryFunc func,
            cl::sycl::buffer<value_type, 1>& out, size_t thread_count,
            cl::sycl::handler &cgh) {

  size_t distance = 128; // ranges::v3::distance(in);
  cl::sycl::nd_range<1> config{distance, cl::sycl::range < 1 > {thread_count}};

  auto wpt = distance / thread_count;

  const auto inIt = in.begin();
  {
    auto outAcc = out.template get_access<cl::sycl::access::mode::write>(cgh);

    // FIXME: this reduction is super slow
    cgh.parallel_for< class ReduceKernel<1, InRng, BinaryFunc> >(config, [=](cl::sycl::nd_item<1> id) {
      auto gid = static_cast<size_t>(id.get_global(0));

      auto start = gid * wpt;

      value_type sum = value_type();

      for (size_t i = 0; i < wpt; ++i)
        sum = func(sum, *(inIt + start + i));

      outAcc[gid] = sum;
    });
  }
}
}
}
}

