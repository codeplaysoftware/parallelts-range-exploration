#pragma once
#include <map>
#include <iostream>
#include <CL/sycl.hpp>
#include <detail/ranges/vector_base.h>
#include <detail/algorithms/transform.h>
#include <detail/algorithms/reduce.h>

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

    // proxy function for calling the transform algorithm
    template<typename InRng1, typename InRng2, typename OutRng, typename BinaryFunc>
    void transform(const InRng1& in1, const InRng2& in2, OutRng& out, BinaryFunc func){

      _queue.submit([&](cl::sycl::handler &cgh) {
        setCGH(cgh); // update the cgh in every vector registerd with this executor
        gstorm::gpu::algorithm::transform(in1, in2, out, func, cgh); // call transform
      });

      _queue.wait_and_throw();

      resetCGH();
    }

    // proxy function for calling the reduce algorithm
    template<typename InRng, typename T, typename BinaryFunc>
    auto reduce(InRng &in, T init, BinaryFunc func){
      using value_type = T;// std::remove_cv_t<std::remove_reference_t<decltype(*in.begin())>>;

      size_t distance = ranges::v3::distance(in);
      size_t thread_count = std::max(128ul, distance / 128ul);

      std::vector<value_type> outVec(thread_count);
      {
        cl::sycl::buffer<value_type, 1> out(outVec.data(), outVec.size());

        _queue.submit([&](cl::sycl::handler &cgh) {
          setCGH(cgh); // update the cgh in every vector registerd with this executor
          gstorm::gpu::algorithm::reduce(in, init, func, out, thread_count, cgh); // call reduce
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
