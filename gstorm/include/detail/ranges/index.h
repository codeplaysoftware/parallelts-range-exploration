//
// Created by mhaidl on 05/07/16.
//

#pragma once

#include <tuple>
#include <iterator>
#include <vector>
#include <detail/traits.h>
#include <type_traits>
#include <PACXX.h>
#include <range/v3/view_facade.hpp>
#include <detail/ranges/vector.h>

namespace gstorm {

  namespace range {

    struct gindex {
    public:
      using size_type = size_t;
      using value_type = std::pair<unsigned int, unsigned int>;
      using reference = value_type&;
      using const_reference = const value_type&;
      using difference_type = std::ptrdiff_t;
      using pointer = value_type*;

      struct iterator : public std::random_access_iterator_tag {
        using size_type = size_t;
        using value_type = std::pair<unsigned int, unsigned int>;
        using reference = value_type&;
        using const_reference = const value_type&;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::random_access_iterator_tag;
        using pointer = value_type*; 

        iterator() = default;

//        reference operator*() { return *it; }

        value_type operator*() const { return std::make_pair(get_global_id(0), get_global_id(1)); }

        iterator& operator++() {
          return *this;
        }

        iterator operator++(int) const {
          return iterator();
        }

        iterator& operator--() {
          return *this;
        }

        iterator operator--(int) const {
          return iterator();
        }

        value_type operator[](difference_type n) { return *(*this); }

        friend iterator advance(const iterator& lhs, difference_type n) {
          return lhs;
        }

        iterator advance(difference_type n) const {
          return *this;
        }

        friend iterator operator+(const iterator& lhs, difference_type n) {
          return iterator();
        }

        friend iterator operator+(difference_type n, const iterator& rhs) {
          return iterator();
        }

        friend iterator operator-(const iterator& lhs, difference_type n) {
          return iterator();
        }

        friend difference_type operator-(const iterator& left, const iterator& right) {
          return 0;
        }

        friend iterator& operator-=(iterator& lhs, difference_type n) {
          return lhs;
        }

        friend iterator& operator+=(iterator& lhs, difference_type n) {
          return lhs;
        }

        friend bool operator<(const iterator& left, const iterator& right) {
          return false;
        }

        friend bool operator>(const iterator& left, const iterator& right) {
          return false;
        }

        friend bool operator<=(const iterator& left, const iterator& right) {
          return false;
        }

        friend bool operator>=(const iterator& left, const iterator& right) {
          return false;
        }

        bool operator==(const iterator& other) const { return true; }

        bool operator!=(const iterator& other) const { return false; }
      };

      using sentinel = iterator;

      gindex() {}

      iterator begin() noexcept { return iterator(); }

      iterator end() noexcept { return iterator(); }

      const iterator begin() const noexcept { return iterator(); }

      const iterator end() const noexcept { return iterator(); }

    };

  }
}
