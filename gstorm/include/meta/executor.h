#pragma once
#include <map>
#include <iostream>
#include <CL/sycl.hpp>
#include <detail/ranges/vector_base.h>
#include <detail/algorithms/transform.h>
#include <detail/algorithms/reduce.h>

#include "aligned_allocator.h"

namespace gstorm
{
  class sycl_exec {
  public:

    // a gvector has to register itselfe with the executor
    // it gets an ID and a pointer to the executor
    // this is needed because gvectors can be moved and they
    // have to update their pointer in the executors gvector map
    void registerGVector(range::gvector_base* ptr) {
      auto pair = std::make_pair(_vectors.size(), ptr);
      _vectors.insert(pair);
      ptr->setID(pair.first);
      ptr->setExecutorPtr(this);
    }

    // when a gvector is moved it updates it pointer in the
    // executors gvector map
    void updateGVector(size_t id, range::gvector_base* ptr){
      _vectors[id] = ptr;
    }

    // proxy function for calling the transform algorithm
    template<typename InRng, typename OutRng, typename UnaryFunc>
    void transform(const InRng& in, OutRng& out, UnaryFunc func){

      _queue.submit([&](cl::sycl::handler &cgh) {
        setCGH(cgh); // update the cgh in every vector registerd with this executor
        gstorm::gpu::algorithm::transform(in, out, func, cgh); // call transform
      });

      _queue.wait_and_throw();

      resetCGH();
    }

    // proxy function for calling the reduce algorithm
    template <typename InRng, typename T, typename BinaryFunc>
    auto reduce(InRng &in, T init, BinaryFunc func) {
      using value_type = T;

      const size_t range_length = ranges::v3::distance(in);

      // Try to access 32K elements per group
      const auto work_per_thread = 256ul;
      const auto local_thread_count = 128ul;
      const auto work_per_group = work_per_thread * local_thread_count;

      const auto group_count =
       range_length % work_per_group == 0
           ? range_length / work_per_group
           : range_length / work_per_group + 1;

      const auto global_thread_count = group_count * local_thread_count;

      const auto elems_in_last_group = range_length % work_per_group;
      const auto threads_writing_in_last_group =
          elems_in_last_group < local_thread_count ? elems_in_last_group
                                                   : local_thread_count;

      const auto threads_writing =
          range_length / work_per_group * local_thread_count +
          threads_writing_in_last_group;

      std::vector<value_type, aligned_allocator<value_type, 4096>> outVec(threads_writing);
      {
        cl::sycl::buffer<value_type, 1> out(outVec.data(), outVec.size(),
                                            {cl::sycl::property::buffer::use_host_ptr{}});

        _queue.submit([&](cl::sycl::handler &cgh) {
          setCGH(cgh); // update the cgh in every vector registerd with this executor
          gstorm::gpu::algorithm::reduce(in, init, func, out, global_thread_count, cgh); // call reduce
        });

        _queue.wait_and_throw();
      }
      resetCGH();
      return  std::accumulate(outVec.begin(), outVec.end(), init, func);
    }

    sycl_exec(cl::sycl::queue _queue) : _queue(_queue) {};

    sycl_exec() {

      auto exception_handler = [] (const cl::sycl::exception_list&) {
        std::abort();
      };

      cl::sycl::cpu_selector device_selector;
      _queue = cl::sycl::queue(device_selector, exception_handler);
      std::cout << "Using device: " << _queue.get_device().get_info<cl::sycl::info::device::name>() << ", from: " << _queue.get_device().get_platform().get_info<cl::sycl::info::platform::name>() << std::endl;
    }
  private:
    // sets the current CGH to all registered vectors
    // TODO: only update used vectors not every vector registerd
    void setCGH(cl::sycl::handler& cgh){
      for(auto ptr : _vectors) {
        ptr.second->setCGH(cgh);
      }
    }

    void resetCGH(){
      for(auto ptr : _vectors) {
        ptr.second->resetCGH();
      }
    }

    cl::sycl::queue _queue;
    std::map<size_t, range::gvector_base*> _vectors;
  };
} // gstorm
