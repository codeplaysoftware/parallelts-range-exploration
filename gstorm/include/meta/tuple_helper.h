//
// Created by mhaidl on 05/07/16.
//

#pragma once
#include <utility>
#include <tuple>

namespace gstorm {
  namespace meta {

// meta programming to extract all values of an
// std::tuple and forward them to a lambad expression

    template<size_t N>
    struct Apply {
      template<typename F, typename T, typename... A>
      static auto apply(F&& f, T&& t, A&& ... a) {
        return Apply<N - 1>::apply(std::forward<F>(f), std::forward<T>(t),
                                   std::get<N - 1>(std::forward<T>(t)),
                                   std::forward<A>(a)...);
      }
    };

    template<>
    struct Apply<0> {
      template<typename F, typename T, typename... A>
      static auto apply(F&& f, T&&, A&& ... a) {
        return std::forward<F>(f)(std::forward<A>(a)...);
      }
    };

    template<typename F, typename T>
    auto apply(F&& f, T&& t) {
      return Apply<std::tuple_size<std::decay_t<T>>
      ::value>::apply(
          std::forward<F>(f), std::forward<T>(t));
    }
/////////////////////////////////////////////////////////////////////////////

// applys a unary lambda function to each element in a tuple
    template<typename F, size_t index>
    struct ForEach {
      template<typename... Ts>
      void operator()(F func, std::tuple<Ts...>& t) {
        ForEach<F, index - 1>()(func, t);
        func(std::get<index - 1>(t));
      }
    };

    template<typename F>
    struct ForEach<F, 0> {
      template<typename... Components>
      void operator()(F func, std::tuple<Components...>& t) { }
    };

    template<class F, class... Ts>
    void for_each_in_tuple(F func, std::tuple<Ts...>& t) {
      ForEach<F, sizeof...(Ts)>()(func, t);
    }
/////////////////////////////////////////////////////////////////////////////
  }
}