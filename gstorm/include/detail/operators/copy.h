//
// Created by mhaidl on 09/08/16.
//

#pragma once

#include <detail/traits.h>
#include <meta/static_const.h>
#include <detail/ranges/vector.h>

namespace gstorm {
namespace gpu {

struct _copy {
  template<typename Exec, typename T>
  auto operator()(Exec &&exec, T &input) const {
    static_assert(traits::is_vector<T>::value, "Only std::vector is currently supported!");
    range::gvector<T> nv(input);
    exec.registerGVector(&nv);
    return nv;
  }
};

template<typename T>
auto operator|(T &lhs, const _copy &rhs) {
  return rhs(lhs);
}

auto copy = gstorm::static_const<_copy>();

}
}