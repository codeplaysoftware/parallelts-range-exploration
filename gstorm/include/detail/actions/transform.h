//
// Created by mhaidl on 10/08/16.
//

#pragma once

#include <PACXX.h>
#include <range/v3/all.hpp>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <detail/operators/copy.h>
#include <detail/ranges/vector.h>
#include <detail/algorithms/transform.h>
#include <meta/static_const.h>

namespace gstorm {
  namespace gpu {
    namespace action {
      template<typename T, typename F>
      struct _transform_action {
        _transform_action(T&& rng, F func) : _rng(std::forward<T>(rng)), _func(func) {}

        auto operator()() {
          algorithm::transform(_rng, _rng, _func);
        }

        operator typename std::remove_reference_t<T>::source_type() {
          operator()();
          return _rng;
        }

      private:
        T _rng;
        F _func;
      };

      template<typename F>
      struct _transform_action_helper {
        _transform_action_helper(F func) : _func(func) {}

        template<typename T>
        auto operator()(T&& rng) const {
          return _transform_action<T, F>(std::forward<T>(rng), _func);
        }

      private:
        F _func;
      };


      struct _transform {
        template<typename F>
        auto operator()(F func) const {
          return _transform_action_helper<decltype(func)>(func);
        }
      };

      auto transform = gstorm::static_const<_transform>();

      template<typename T, typename F>
      auto operator|(range::gvector<T>&& lhs, _transform_action_helper<F>&& rhs) {
        return rhs(std::forward<decltype(lhs)>(lhs));
      }

      template<typename Rng, typename F>
      auto operator|(Rng& lhs, _transform_action_helper<F>&& rhs) {
        // evaluate explicitly here because _gpu_copy is destroyed before it collapses into its source type
        auto gpu_copy = lhs | gpu::copy;
        // auto trans =
        return rhs(std::move(gpu_copy));;
      };

    }
  }
}
