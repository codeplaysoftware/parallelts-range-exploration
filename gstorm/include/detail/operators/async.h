//
// Created by mhaidl on 16/08/16.
//

#pragma once

#include <meta/static_const.h>
#include <range/v3/all.hpp>
#include <vector>
#include <future>
#include <iostream>
#include <detail/ranges/vector.h>
#include <detail/algorithms/transform.h>
#include <meta/tuple_helper.h>

namespace gstorm {
  namespace gpu {

    namespace meta {

      template<typename T, std::enable_if_t<traits::is_vector<T>::value>* = nullptr>
      auto translate_memory(const T& data) {
        return range::gvector<T>(data);
      }

      template<typename T, std::enable_if_t<traits::is_gvector<T>::value>* = nullptr>
      const auto& translate_memory(const T& data) {
        return data;
      }

      template<typename T, std::enable_if_t<!traits::is_vector<T>::value && !traits::is_gvector<T>::value>* = nullptr>
      const auto& translate_memory(const T& data) {
        return data;
      }

    }

    template<typename T>
    struct _async {
      using type_ = std::remove_reference_t<std::remove_cv_t<T>>;
      using value_type = decltype(*std::declval<type_>().begin());

      _async(T view) : _view(view) {}

      auto operator()() {

        // get a BindingPromise instance from PACXX
        // the PACXX runtime ensure that the promise is alive when the callback is fired
        auto& promise = pacxx::v2::get_executor().getPromise<range::gvector<std::vector<value_type>>>(
            ranges::v3::distance(_view));
        auto future = promise.getFuture(); // get an std::future from the promise
        auto& outRng = promise.getBoundObject(); // get the bound object that will survive until the callback fires

        gpu::algorithm::transform(_view, outRng, [](auto&& in) { return in; },
                                  [&]() mutable {
                                    promise.fulfill(); // at this point the computation is finished we can fulfill the promise
                                    pacxx::v2::get_executor().forgetPromise(promise);
                                  });

        return future;
      }


    private:
      T _view;
    };

    struct _async_helper {
      template<typename T, typename = std::enable_if_t<ranges::v3::is_view<T>::value>>
      auto operator()(T&& view) const { return _async<T>(std::forward<T>(view)); }

      template<typename F, typename... Ts, typename = std::enable_if_t<!ranges::v3::is_view<F>::value>>
      auto operator()(F&& function, const Ts& ... args) const {

        auto tpl = std::tuple<decltype(meta::translate_memory(args))...>(meta::translate_memory(args)...);

        auto view = gstorm::meta::apply(function, tpl);
        _async<decltype(view)> kernel(view);

        auto future = kernel();
        return future;
      };
    };

    auto async = gstorm::static_const<_async_helper>();

    template<typename View>
    auto operator|(View&& lhs, _async_helper&& rhs) {
      return rhs(lhs);
    };

  }
}