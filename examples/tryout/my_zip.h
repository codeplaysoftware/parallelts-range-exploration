#pragma once

#include <gstorm.h>
#include <tuple>
#include <utility>
#include <type_traits>

/**
 * Workaround for creating a tuple using gvector.
 *
 * Fields in std::tuple cannot be references to global memory.
 * Have to use cl::sycl::global_ptr<T> instead for elements of gvector.
 * Views will behave as before.
 *
 * Ts() will choose the correct type to use for the tuple.
 */
template<typename... Ts>
struct make_tuple_for_zip {
  auto operator()(typename Ts::fn_src_type&&... args) const {
    return std::make_tuple(Ts{}(std::forward<typename Ts::fn_src_type>(args))...);
  }
};

/**
 * Create a tuple element, will be of type T
 */
template<typename T, class Enable = void>
struct make_element_for_tuple {

  using src_type = ranges::range_value_t<std::remove_reference_t<T>>;
  using fn_src_type = src_type&&;
  using dest_type = src_type;

  dest_type& operator()(src_type&& src) {
    return src;
  }
};

/**
 * Create a tuple element for gvector, has to be cl::sycl::global_ptr<T> instead of __global T&
 */
template<typename T>
struct make_element_for_tuple<T, std::enable_if_t<gstorm::traits::is_gvector<std::remove_reference_t<T>>::value>> {

  using src_type = ranges::range_value_t<std::remove_reference_t<T>>;
  using fn_src_type = src_type&;
  using dest_type = cl::sycl::global_ptr<src_type>;

  dest_type operator()(src_type& src) {
    return dest_type(&src);
  }
};

struct my_zip_fn {

  /**
   * Workaround for zip using gvector.
   *
   * Fields in std::tuple cannot be references to global memory.
   * Have to use cl::sycl::global_ptr<T> instead for elements of gvector.
   * Views will behave as before.
   *
   * make_element_for_tuple<Rngs>... will choose the correct type
   * to use for each input range.
   */
  template<typename...Rngs>
  auto operator()(Rngs &&... rngs) const {
    return ranges::view::zip_with(
        make_tuple_for_zip<make_element_for_tuple<Rngs>...>{},
        std::forward<Rngs>(rngs)...);
  }
};

inline namespace {
using ranges::v3::static_const;
RANGES_INLINE_VARIABLE(my_zip_fn, my_zip)
}
