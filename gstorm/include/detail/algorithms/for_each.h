//
// Created by mhaidl on 27/08/16.
//

#pragma once

#include <PACXX.h>
#include <range/v3/all.hpp>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <detail/operators/copy.h>
#include <detail/ranges/vector.h>
#include <meta/static_const.h>

#include <detail/common/Meta.h>

namespace gstorm {
  namespace gpu {
    namespace algorithm {

      template<typename InTy, typename UnaryFunc>
      struct for_each_functor {
        void operator()(InTy in,
                        size_t distance,
                        UnaryFunc func) const {
          auto id = get_global_id(0);
          if (static_cast<size_t>(id) >= distance) return;

          func(*(in + id));
        }
      };

      template<typename InRng, typename UnaryFunc>
      auto for_each(InRng&& in, UnaryFunc&& func) {
        constexpr size_t thread_count = 128;

        auto distance = ranges::v3::distance(in);

        auto kernel = pacxx::v2::kernel(
            for_each_functor<decltype(in.begin()), pacxx::meta::callable_wrapper<UnaryFunc>>(),
            {{(distance + thread_count - 1) / thread_count},
             {thread_count}, 0, 0});

        kernel(in.begin(), distance, pacxx::meta::callable_wrapper<UnaryFunc>(func));
      };
    }
  }
}
