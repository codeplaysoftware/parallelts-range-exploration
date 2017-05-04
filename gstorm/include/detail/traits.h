//
// Created by mhaidl on 05/07/16.
//

#pragma once

#include <detail/ranges/vector.h>


namespace gstorm {
  namespace traits {

    template<class T>
    struct is_vector : std::false_type {
    };

    template<class T, class Alloc>
    struct is_vector<std::vector<T, Alloc>> : std::true_type {
    };

    template<class T>
    struct is_gvector : std::false_type {
    };

    template<template<typename> class T, typename U>
    struct is_gvector<T<U>> : std::is_same<T<U>, range::gvector<U>> {
    };



    template<template<typename...> class Template, typename T>
    struct is_specialization_of : std::false_type {
    };

    template<template<typename...> class Template, typename... Args>
    struct is_specialization_of<Template, Template<Args...>> : std::true_type {
    };

    template<typename T>
    struct void_ {
      typedef void type;
    };

    template<typename T, typename = void>
    struct is_constructable : std::false_type {
    };

    template<typename T>
    struct is_constructable<T, typename void_<typename T::construction_type>::type> : std::true_type {
    };


    template<typename T, bool = is_constructable<T>::value>
    struct view_traits {
    };

    template<typename T>
    struct view_traits<T, true> {
      using construction_type = typename T::construction_type;
      static const unsigned arity = std::tuple_size<construction_type>::value;
    };

    template<typename T>
    struct view_traits<T, false> {
      using construction_type = void;
      static const unsigned arity = 0;
    };
  }
}