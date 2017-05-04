//
// Created by mhaidl on 23/08/16.
//

#pragma once

#include <cstddef>
#include <range/v3/all.hpp>
#include <PACXX.h>
#include <detail/ranges/vector.h>
#include <detail/common/Timing.h>

namespace gstorm {

namespace meta {
template<int First, int Last, typename Fn>
struct _static_for {
  _static_for(Fn f) : func(f) {}

  template<typename T>
  auto operator()(T init, T y) const {
    return _static_for<First + 1, Last, Fn>(func)(init, func(init, y));
  }

  Fn func;
};

template<int N, typename Fn>
struct _static_for<N, N, Fn> {
  _static_for(Fn f) {}

  template<typename T>
  auto operator()(T init, T y) const {
    return y;
  }
};

template<int First, int Last, typename Fn>
auto static_for(Fn f) {
  return _static_for<First, Last, Fn>(f);
}

}

namespace gpu {
namespace algorithm {
namespace detail {

template<typename InTy, typename OutTy, typename BinaryFunc>
struct reduce_functorGPUNvidia {
  using value_type = std::remove_reference_t<decltype(*std::declval<InTy>())>;
private:
  BinaryFunc func;
public:

  reduce_functorGPUNvidia(BinaryFunc &&f) : func(f) {}

  void operator()(InTy in, OutTy out, size_t distance, size_t ept) const {
    pacxx::v2::shared_memory <value_type> sdata;
    size_t tid = get_local_id(0);

    auto n = distance;
    int elements_per_thread = pacxx::v2::_stage([&] { return ept; });

    value_type sum = 0;

    size_t gridSize = get_local_size(0) * get_num_groups(0);
    size_t i = get_global_id(0);

    for (int x = 0; x < elements_per_thread; ++x) {
      sum = func(sum, *(in + i));
      i += gridSize;
    }

    while (i < n) {
      sum = func(sum, *(in + i));
      i += gridSize;
    }

    sdata[tid] = sum;
    barrier(1);
    if (get_local_size(0) >= 1024) {
      if (tid < 512)
        sdata[tid] = func(sdata[tid], sdata[tid + 512]);
      barrier(1);
    }
    if (get_local_size(0) >= 512) {
      if (tid < 256)
        sdata[tid] = func(sdata[tid], sdata[tid + 256]);
      barrier(1);
    }
    if (get_local_size(0) >= 256) {
      if (tid < 128)
        sdata[tid] = func(sdata[tid], sdata[tid + 128]);
      barrier(1);
    }
    if (tid < 64)
      sdata[tid] = func(sdata[tid], sdata[tid + 64]);
    barrier(1);


    if(tid < 32)
      sdata[tid] = func(sdata[tid], sdata[tid + 32]);
    barrier(1);

    if(tid < 16)
      sdata[tid] = func(sdata[tid], sdata[tid + 16]);
    barrier(1);

    if(tid < 8)
      sdata[tid] = func(sdata[tid], sdata[tid + 8]);
    barrier(1);

    if(tid < 4)
      sdata[tid] = func(sdata[tid], sdata[tid + 4]);
    barrier(1);

    if(tid < 2)
      sdata[tid] = func(sdata[tid], sdata[tid + 2]);
    barrier(1);

    if(tid < 1)
      sdata[tid] = func(sdata[tid], sdata[tid + 1]);
    barrier(1);

    if (tid == 0)
      *(out + static_cast<size_t>(get_group_id(0))) = sdata[tid];

//    if (tid < 32) {
//      volatile value_type *sm = &sdata[0];
//      sm[tid] = func(sm[tid], sm[tid + 32]);
//      sm[tid] = func(sm[tid], sm[tid + 16]);
//      sm[tid] = func(sm[tid], sm[tid + 8]);
//      sm[tid] = func(sm[tid], sm[tid + 4]);
//      sm[tid] = func(sm[tid], sm[tid + 2]);
//      sm[tid] = func(sm[tid], sm[tid + 1]);
//    }
//    if (tid == 0)
//      *(out + static_cast<size_t>(get_group_id(0))) = sdata[tid];
  }
};

template<typename InTy, typename OutTy, typename BinaryFunc>
struct reduce_functorCPU {
  using value_type = std::remove_reference_t<decltype(*std::declval<InTy>())>;
private:
  BinaryFunc func;
public:

  reduce_functorCPU(BinaryFunc &&f) : func(f) {}

  void operator()(InTy in, OutTy out, size_t wpt) const {

    value_type sum = 0;

    const auto tid = get_local_id(0);
    const auto gid = get_group_id(0);
    const auto gsize = get_local_size(0);
    size_t i = tid * wpt + gid * gsize * wpt;

    for (size_t j = 0; j < wpt; ++j) {
      sum = func(sum, *(in + i + j));
    }

    *(out + static_cast<size_t>(get_global_id(0))) = sum;

  }
};

template<typename InRng, typename BinaryFunc>
auto reduceGPUNvidia(InRng &&in, std::remove_reference_t<decltype(*in.begin())> init, BinaryFunc &&func) {
  auto result_val = init;

  size_t distance = ranges::v3::distance(in);
  size_t thread_count = 128;
  if (distance >= 1 << 24)
    thread_count = 1024;
  thread_count = std::min(thread_count, distance);
  size_t ept = 1;
  if (distance > thread_count * 2) {
    do {
      ept *= 2;
    } while (distance / (thread_count * ept) > 130);
  }

  size_t block_count = std::max(distance / thread_count + (distance % thread_count > 0 ? 1 : 0), 1ul);
  const size_t max_blocks = pacxx::v2::Executor::get(0).getConcurrentCores() * 10;
  block_count = std::min(block_count, max_blocks);
  ept = distance / (thread_count * block_count);

  using value_type = std::remove_reference_t<decltype(*in.begin())>;
  std::vector<value_type> result(block_count, value_type());
  range::gvector <std::vector<value_type>> out(result);

  using FunctorTy = reduce_functorGPUNvidia<decltype(in.begin()), decltype(out.begin()), BinaryFunc>;

  auto kernel = pacxx::v2::kernel<FunctorTy, pacxx::v2::Target::GPU>(
                              FunctorTy(std::forward<BinaryFunc>(func)),
                              {{block_count}, {thread_count}, 0, thread_count * sizeof(value_type)});

  kernel(in.begin(), out.begin(), distance, ept);

  result = out;

  result_val = ranges::v3::accumulate(result, init, func);

  return result_val;
};

template<typename InRng, typename BinaryFunc>
auto reduceCPU(InRng &&in, std::remove_reference_t<decltype(*in.begin())> init, BinaryFunc &&func) {
  auto result_val = init;
  using value_type = decltype(init);
  auto& exec = pacxx::v2::Executor::get(0);

  size_t distance = ranges::v3::distance(in);

  size_t thread_count = exec.getVectorizationWidth<value_type>();
  size_t block_count = exec.getConcurrentCores();

  std::vector<value_type> result(block_count * thread_count, value_type());
  range::gvector <std::vector<value_type>> out(result);

  size_t wpt = distance / (block_count * thread_count);
  bool remainder = distance % (block_count * thread_count) != 0;


  using FunctorTy = reduce_functorCPU<decltype(in.begin()), decltype(out.begin()), BinaryFunc>;

  auto kernel = pacxx::v2::kernel<FunctorTy, pacxx::v2::Target::CPU>(
                                  FunctorTy(std::forward<BinaryFunc>(func)),
                                  {{block_count}, {thread_count}, 0, 0});

  kernel(in.begin(), out.begin(), wpt);

  result = out;

  result_val = ranges::v3::accumulate(result, init, func);

  if (remainder) {
    auto rem = ranges::view::counted(in.begin() + wpt * block_count * thread_count, distance - (wpt * block_count * thread_count));
    result_val = ranges::v3::accumulate(rem, result_val, func);
  }

  return result_val;
};

}

template<typename InRng, typename BinaryFunc>
auto reduce(InRng &&in, std::remove_reference_t<decltype(*in.begin())> init, BinaryFunc &&func) {
  using namespace pacxx::v2;
  auto &exec = Executor::get(0); // get default executor

  switch (exec.getExecutingDeviceType()) {
  case ExecutingDevice::GPUNvidia:return detail::reduceGPUNvidia(std::forward<InRng>(in), init, std::forward<BinaryFunc>(func));
  case ExecutingDevice::CPU:return detail::reduceCPU(std::forward<InRng>(in), init, std::forward<BinaryFunc>(func));
  }

  return init;
}
}
}
}

