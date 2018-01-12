//
// Created by mhaidl on 23/08/16.
//

#pragma once

#include <range/v3/all.hpp>
#include <iostream>
#include <CL/sycl.hpp>

namespace gstorm {
namespace gpu {
namespace algorithm {

template <int x, typename... Ts>
class ReduceKernel{};

template<typename InRng, typename T, typename BinaryFunc, typename value_type = T>
auto reduce(InRng &in, T init, BinaryFunc func,
            cl::sycl::buffer<value_type, 1>& out, size_t thread_count,
            cl::sycl::handler &cgh) {

  size_t distance = ranges::v3::distance(in);
  cl::sycl::nd_range<1> config{thread_count, 128ul};

  const auto inIt = in.begin();
  {
    auto outAcc = out.template get_access<cl::sycl::access::mode::discard_write>(cgh);

    cgh.parallel_for< class ReduceKernel<1, BinaryFunc> >(config, [=](cl::sycl::nd_item<1> id) {
      auto global_id = id.get_global(0);
      auto local_id = id.get_local(0);
      auto group_id = id.get_group(0);
      auto num_groups = id.get_num_groups(0);
      auto local_size = id.get_local_range(0);

      auto elems_per_group = distance / num_groups;

      auto start = inIt + elems_per_group * group_id + local_id;

      value_type sum = value_type();

      for (size_t i = 0; i < elems_per_group; i += local_size)
        sum = func(sum, *(start + i));

      outAcc[global_id] = sum;
    });
  }
}
}
}
}

