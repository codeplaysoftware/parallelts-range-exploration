//
// Created by mhaidl on 30/08/16.
//

#pragma once

#include <range/v3/all.hpp>
#include <detail/operators/functional.h>
#include <detail/algorithms/transform.h>

namespace gstorm {
  namespace gpu {
    namespace algorithm {

      template<typename InRng, typename OutRng, typename UnaryFunc>
      void transform(InRng&& in, OutRng& out, UnaryFunc&& func);

      template<typename OutRng, typename T>
      void fill(OutRng& out, T v) {

        auto distance = ranges::v3::distance(out);
        auto in = ranges::v3::view::repeat(v) | ranges::v3::view::take(distance);

        gstorm::gpu::algorithm::transform(in, out, functional::identity());
      }
    }
  }
}