#pragma once
#include <map>
#include <CL/sycl.hpp>
#include <detail/ranges/vector_base.h>
#include <detail/algorithms/transform.h>


namespace gstorm
{

  class sycl_exec {
  public:

    void registerGVector(range::gvector_base* ptr) {
      auto pair = std::make_pair(_vectors.size(), ptr);
      _vectors.insert(pair);
      ptr->setID(pair.first);
      ptr->setExecutorPtr(this);
    }

    void updateGVector(size_t id, range::gvector_base* ptr){
      _vectors[id] = ptr;
    }

    template<typename InRng, typename OutRng, typename UnaryFunc>
    void transform(const InRng& in, OutRng& out, UnaryFunc func){

      _queue.submit([&](cl::sycl::handler &cgh) {
        setCGH(cgh);
        gstorm::gpu::algorithm::transform(in, out, func, cgh);
      });

    }

  private:

    // sets the current CGH to all registered vectors
    // TODO: only update used vectors not every vector registerd
    void setCGH(cl::sycl::handler& cgh){
      for(auto ptr : _vectors) {
        ptr.second->setCGH(cgh);
      }
    }

    cl::sycl::queue _queue;
    std::map<size_t, range::gvector_base*> _vectors;
  };
}