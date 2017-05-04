//
// Created by mhaidl on 05/07/16.
//

#pragma once

#include <tuple>
#include <iterator>
#include <iostream>
#include <vector>
#include <detail/traits.h>
#include <type_traits>
#include <memory>
#include <CL/sycl.hpp>
#include <range/v3/view_facade.hpp>
#include <detail/algorithms/transform.h>
//#include <detail/algorithms/fill.h>
// this is a copy from std/experimental to get a version of
// optional which does not throw and does not call abort
// on the device side
// TODO: make a own optional implementation that does nocht
//       belong to the STL
#include "../../meta/optional"
#include <detail/ranges/vector_base.h>
#include "../../meta/executor.h"

namespace gstorm {

namespace traits {
template<typename T>
struct range_forward_traits {
  using base_type = T;

  using size_type = typename T::size_type;
  using value_type = typename T::value_type;
  using reference = typename std::conditional<std::is_const<T>::value,
                                              const typename T::reference,
                                              typename T::reference>::type;
  using const_reference = typename T::const_reference;
  using difference_type = typename T::difference_type;
};
}


namespace range{
template<typename T>
struct gvector : public traits::range_forward_traits<T>, public range::gvector_base {
public:
  using source_type = T;
  using size_type = typename T::size_type;
  using value_type = typename T::value_type;
  using reference = typename std::conditional<std::is_const<T>::value,
                                              const typename T::reference,
                                              typename T::reference>::type;
  using const_reference = typename T::const_reference;
  using difference_type = typename T::difference_type;
  using pointer = value_type *;


  struct iterator : public std::random_access_iterator_tag, public traits::range_forward_traits<T> {
    using size_type = typename T::size_type;
    using value_type = typename T::value_type;
    using reference = typename std::conditional<std::is_const<T>::value,
                                                const typename T::reference,
                                                typename T::reference>::type;
    using const_reference = typename T::const_reference;
    using difference_type = typename T::difference_type;
    using iterator_category = std::random_access_iterator_tag;
    using pointer = value_type *;
    using sycl_accessor_type = cl::sycl::accessor<value_type, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer>;

    iterator(size_t offset = 0, ptrdiff_t ref_back = 0) : accessor(), owner(ref_back), it(offset), _id(99999) {
#ifndef __SYCL_DEVICE_ONLY__
      if (owner != 0)
        _id = (reinterpret_cast<gvector*>(owner))->registerIterator(*this); // this should never be executed on the device
#endif
    }

    ~iterator(){
#ifndef __SYCL_DEVICE_ONLY__
      if (owner != 0 && _id != 99999)
        (reinterpret_cast<gvector*>(owner))->forgetIterator(_id); // this should never be executed on the device
#endif
    }

    iterator(const iterator& other) : owner(other.owner), it(other.it), _id(99999){
      if (other.hasAccessor())
        accessor.emplace(other.accessor.value());
      else {
#ifndef __SYCL_DEVICE_ONLY__
        if (owner != 0)
          _id = (reinterpret_cast<gvector*>(owner))->registerIterator(*this); // this should never be executed on the device
#endif
      }
    }
    //iterator(iterator&& other) : accessor(std::move(other.accessor)), it(other.it){ }

    bool hasAccessor() const { return static_cast<bool>(accessor); }

    sycl_accessor_type getAccessor() const { return accessor.value(); }

    void setAccessor(sycl_accessor_type acc) { accessor.emplace(acc); }

    explicit iterator(sycl_accessor_type acc, size_t offset = 0, ptrdiff_t ref_back = 0) : accessor(acc),
                                                                                                owner(ref_back), it(offset), _id(99999) {
    }

    reference operator*() {
      return accessor.value()[it];
    }

    reference operator*() const {
      return accessor.value()[it];
    }

    iterator &operator++() {
      ++it;
      return *this;
    }

    iterator operator++(int) const {
      auto ip = it;
      ++it;
      if (!hasAccessor())
        return iterator(ip);
      return iterator(accessor.value(), ip, owner);
    }

    iterator &operator--() {
      --it;
      return *this;
    }

    iterator operator--(int) const {
      auto ip = it;
      --it;
      if (!hasAccessor())
        return iterator(ip);
      return iterator(accessor.value(), ip, owner);
    }

    reference operator[](difference_type n) { return accessor.value()[it + n]; }

    friend iterator advance(const iterator &lhs, difference_type n) {
      return lhs.advance(n);
    }

    iterator advance(difference_type n) const {
      it += n;
      return *this;
    }

    friend iterator operator+(const iterator &lhs, difference_type n) {
      if (!lhs.hasAccessor())
        return iterator(lhs.it + n);
      return iterator(lhs.accessor.value(), lhs.it + n, lhs.owner);
    }

    friend iterator operator+(difference_type n, const iterator &rhs) {
      if (!rhs.hasAccessor())
        return iterator(rhs.it + n);
      return iterator(rhs.accessor.value(), rhs.it + n, rhs.owner);
    }

    friend iterator operator-(const iterator &lhs, difference_type n) {
      if (!lhs.hasAccessor())
        return iterator(lhs.it - n);
      return iterator(lhs.accessor.value(), lhs.it - n, lhs.owner);
    }

    friend difference_type operator-(const iterator &left, const iterator &right) {
      return left.it - right.it;
    }

    friend iterator &operator-=(iterator &lhs, difference_type n) {
      lhs.it -= n;
      return lhs;
    }

    friend iterator &operator+=(iterator &lhs, difference_type n) {
      lhs.it += n;
      return lhs;
    }

    friend bool operator<(const iterator &left, const iterator &right) {
      return left.it < right.it;
    }

    friend bool operator>(const iterator &left, const iterator &right) {
      return left.it > right.it;
    }

    friend bool operator<=(const iterator &left, const iterator &right) {
      return left.it <= right.it;
    }

    friend bool operator>=(const iterator &left, const iterator &right) {
      return left.it >= right.it;
    }

    bool operator==(const iterator &other) const { return other.it == it; }

    bool operator!=(const iterator &other) const { return other.it != it; }

  private:
    mutable std::experimental::optional<sycl_accessor_type> accessor;
    mutable ptrdiff_t owner;
    mutable size_t it;
    mutable size_t _id;
  };

  using sentinel = iterator;

  gvector() : gvector_base(), _buffer(nullptr), _size(0)  {}

  gvector(size_t size) : gvector_base(), _buffer(new cl::sycl::buffer<typename T::value_type>(size)),
                         _size(size) {

  }

  gvector(size_t size, value_type value) : gvector_base(), _buffer(new cl::sycl::buffer<typename T::value_type>(size)),
                                           _size(size) {
//    gpu::algorithm::fill(*this, value);
  }

  gvector(T &vec) : gvector_base(), _buffer(new cl::sycl::buffer<typename T::value_type>(vec.data(), vec.size())),
                    _size(vec.size())  {
  }

  ~gvector() {}

  gvector(const gvector &src) = delete;

  gvector(gvector &&other) {
    _buffer = std::move(other._buffer);
    other._buffer.reset(nullptr);
    _cgh = other._cgh;
    other._cgh = nullptr;
    _size = other._size;
    _id = other._id;
    other._size = 0;
    _exec = other._exec;
    _exec->updateGVector(_id, this);
  }

  gvector &operator=(gvector &&other) {
    _buffer = std::move(other._buffer);
    other._buffer.reset(nullptr);
    _cgh = other._cgh;
    other._cgh = nullptr;
    _size = other._size;
    _id = other._id;
    other._size = 0;
    _exec = other._exec;
    _exec->updateGVector(_id, this);
    return *this;
  }

  iterator begin() noexcept {
    if (_cgh)
      return iterator(_buffer->template get_access<cl::sycl::access::mode::read_write>(*_cgh), 0, reinterpret_cast<ptrdiff_t>(this));
    return iterator(0, reinterpret_cast<ptrdiff_t>(this));
  }

  iterator end() noexcept {
    if (_cgh)
      return iterator(_buffer->template get_access<cl::sycl::access::mode::read_write>(*_cgh), _size, reinterpret_cast<ptrdiff_t>(this));
    return iterator(_size, reinterpret_cast<ptrdiff_t>(this));
  }

  const iterator begin() const noexcept {
    if (_cgh)
      return iterator(_buffer->template get_access<cl::sycl::access::mode::read_write>(*_cgh), 0, reinterpret_cast<ptrdiff_t>(this));
    return iterator(0, reinterpret_cast<ptrdiff_t>(this));
  }

  const iterator end() const noexcept {
    if (_cgh)
      return iterator(_buffer->template get_access<cl::sycl::access::mode::read_write>(*_cgh), _size, reinterpret_cast<ptrdiff_t>(this));
    return iterator(_size, reinterpret_cast<ptrdiff_t>(this));
  }

  size_t size() const { return _size; }

  void resize(size_type size) {
//    auto new_buffer = new cl::sycl::buffer<typename T::value_type>(size);
//    if (_buffer)
//      _buffer->copyTo(new_buffer->get());
//    _size = size;
//    _buffer.reset(new_buffer);
  }

  operator typename std::remove_cv<T>::type() const {
    T tmp(_size);
    // download data to tmp
    return tmp;
  }

  void swap(gvector &other) {
    std::swap(_buffer, other._buffer);
    std::swap(_size, other._size);
  }

  typename T::value_type *data() {
    auto accesor =
        _buffer->template get_access<cl::sycl::access::mode::read_write, cl::sycl::access::target::host_buffer>();
    return &accesor[0];
  }

  const typename T::value_type *data() const {
    auto accesor =
        _buffer->template get_access<cl::sycl::access::mode::read_write, cl::sycl::access::target::host_buffer>();
    return &accesor[0];
  }

  reference operator[](difference_type n) const {
    auto accesor =
        _buffer->template get_access<cl::sycl::access::mode::read_write, cl::sycl::access::target::host_buffer>();
    return accesor[n];
  }

  auto registerIterator(iterator& it) {
    auto pair = std::make_pair(_iterators.size(), &it);
    _iterators.insert(pair);
    return pair.first;
  }

  void forgetIterator(size_t id){
    _iterators[id] = nullptr;
  }

  virtual void updateAccessors() override {
    auto acc = _buffer->template get_access<cl::sycl::access::mode::read_write>(*_cgh);
    for (auto& p : _iterators) {
      if (p.second) {
        p.second->setAccessor(acc);
      }
    }
  }

private:
  std::unique_ptr<cl::sycl::buffer < typename T::value_type>> _buffer;
  std::map<size_t, iterator*> _iterators;
  size_t _size;
};

template<typename T>
gvector<T> gpu_vector(T &vec) {
  return gvector<T>(vec);
}
}
}
