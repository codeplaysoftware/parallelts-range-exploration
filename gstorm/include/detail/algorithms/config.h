//
// Created by mhaidl on 23/08/16.
//

#pragma once


#include <range/v3/all.hpp>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <detail/operators/copy.h>
#include <detail/ranges/vector.h>
#include <meta/static_const.h>

namespace gstorm {
  namespace gpu {
    namespace algorithm {
    
      struct dim3{
        unsigned x, y, z; 
        dim3(unsigned x = 1, unsigned y = 1, unsigned z = 1) : x(x), y(y), z(z) {}
      };

      struct config{
        dim3 blocks, threads;
        config(dim3 b = {1, 1, 1}, dim3 t = {1, 1, 1}) : blocks(b), threads(t) {}
        
      };

    }
  }
}
