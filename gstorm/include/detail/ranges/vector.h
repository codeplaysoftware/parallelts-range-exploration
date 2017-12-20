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
#include <limits>
#include <CL/sycl.hpp>
#include <range/v3/view_facade.hpp>
#include <detail/algorithms/transform.h>
#include <detail/ranges/vector_base.h>
#include <meta/executor.h>

namespace gstorm {

namespace traits {
template<typename T>
struct range_forward_traits {
  using base_type = T;

  using size_type = typename T::size_type;
  using value_type = typename T::value_type;
  using reference = std::conditional_t<std::is_const<T>::value,
                                       const typename T::reference,
                                       typename T::reference>;
  using const_reference = typename T::const_reference;
  using difference_type = typename T::difference_type;
};
}


// this is the main container used in sycl parallel algorihtms

namespace range{

constexpr auto size_t_max = std::numeric_limits<size_t>::max();

template<typename T>
struct gvector;

template<typename T>
struct iterator : public std::random_access_iterator_tag, public traits::range_forward_traits<T> {
  using size_type = typename T::size_type;
  using value_type = typename T::value_type;
  using reference = std::conditional_t<std::is_const<T>::value,
                                       const typename T::reference,
                                       typename T::reference>;
  using const_reference = typename T::const_reference;
  using difference_type = typename T::difference_type;
  using iterator_category = std::random_access_iterator_tag;
  using pointer = value_type *;

  // define the accessor's type
  using sycl_accessor_type = cl::sycl::accessor<value_type, 1,
                                                cl::sycl::access::mode::read_write,
                                                cl::sycl::access::target::global_buffer,
                                                cl::sycl::access::placeholder::true_t>;

  iterator(size_t offset = 0, ptrdiff_t ref_back = 0) : accessor(), owner(ref_back), it(offset), id_(size_t_max) {
#ifndef __SYCL_DEVICE_ONLY__
    if (owner)
      id_ = (reinterpret_cast<gvector<T>*>(owner))->registerIterator(*this); // this should never be executed on the device
#endif
  }

  ~iterator(){
#ifndef __SYCL_DEVICE_ONLY__
    if (owner && id_ != size_t_max)
      (reinterpret_cast<gvector<T>*>(owner))->forgetIterator(id_); // this should never be executed on the device
#endif
  }

  iterator(const iterator& other) : has_accessor(other.has_accessor), accessor(other.accessor), owner(other.owner), it(other.it), id_(){
    if (!other.hasAccessor()) {
#ifndef __SYCL_DEVICE_ONLY__
      if (owner)
        id_ = (reinterpret_cast<gvector<T>*>(owner))->registerIterator(*this); // this should never be executed on the device
#endif
    }
  }

  bool hasAccessor() const { return has_accessor; }

  sycl_accessor_type getAccessor() const { return accessor; }

  void setAccessor(sycl_accessor_type acc) { has_accessor = true; accessor = acc; }

  explicit iterator(sycl_accessor_type acc, size_t offset = 0, ptrdiff_t ref_back = 0) :
    has_accessor(true),
    accessor(acc),
    owner(ref_back),
    it(offset),
    id_(size_t_max) {}

  reference operator*() {
    return accessor[it];
  }

  reference operator*() const {
    return accessor[it];
  }

  iterator &operator++() {
    ++it;
    return *this;
  }

  iterator operator++(int) {
    auto ip = it;
    ++it;
    return iterator(accessor, ip, owner);
  }

  iterator &operator--() {
    --it;
    return *this;
  }

  iterator operator--(int) {
    auto ip = it;
    --it;
    return iterator(accessor, ip, owner);
  }

  reference operator[](difference_type n) { return accessor[it + n]; }

  friend iterator advance(const iterator &lhs, difference_type n) {
    return lhs.advance(n);
  }

  iterator advance(difference_type n) {
    it += n;
    return *this;
  }

  friend iterator operator+(const iterator &lhs, difference_type n) {
    return iterator(lhs.accessor, lhs.it + n, lhs.owner);
  }

  friend iterator operator+(difference_type n, const iterator &rhs) {
    return iterator(rhs.accessor, rhs.it + n, rhs.owner);
  }

  friend iterator operator-(const iterator &lhs, difference_type n) {
    return iterator(lhs.accessor, lhs.it - n, lhs.owner);
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
    return right < left;
  }

  friend bool operator<=(const iterator &left, const iterator &right) {
    return left < right || left == right;
  }

  friend bool operator>=(const iterator &left, const iterator &right) {
    return right < left || right == left;
  }

  bool operator==(const iterator &other) const { return other.it == it; }

  bool operator!=(const iterator &other) const { return !(other == *this); }

private:
  bool has_accessor = false;
  sycl_accessor_type accessor;
  ptrdiff_t owner;
  size_t it;
  size_t id_;
};


template<typename T>
struct gvector : public traits::range_forward_traits<T>, public range::gvector_base {
public:
  using source_type = T;
  using size_type = typename T::size_type;
  using value_type = typename T::value_type;
  using reference = std::conditional_t<std::is_const<T>::value,
                                       const typename T::reference,
                                       typename T::reference>;
  using const_reference = typename T::const_reference;
  using difference_type = typename T::difference_type;
  using pointer = value_type *;

  using sycl_accessor_type = cl::sycl::accessor<value_type, 1,
                                                cl::sycl::access::mode::read_write,
                                                cl::sycl::access::target::global_buffer,
                                                cl::sycl::access::placeholder::true_t>;


  using sentinel = iterator<T>;

  friend struct iterator<T>;

  gvector() : gvector_base(), _buffer(nullptr), _size(0) {}

  gvector(size_t size) : gvector_base(),
                         _buffer(new cl::sycl::buffer<value_type>(size)),
                         _size(size) {

  }

  gvector(size_t size, value_type value) : gvector_base(),
                                           _buffer(new cl::sycl::buffer<value_type>(size)),
                                           _size(size) {
    // TODO: call a fill algorithm to initalize the gvector to respect RAII
  }

  gvector(T &vec) : gvector_base(),
                    _buffer(new cl::sycl::buffer<value_type>(vec.data(), vec.size())),
                    _size(vec.size())  {}

  ~gvector() {}

  gvector(const gvector &src) = delete; // copying a gvector is not allowed yet


  // move ctor for the gvector class
  // the move ctor will update to the new pointer by its bound executor
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

  // move assignment for the gvector class
  // the move assignment will update to the new pointer by its bound executor
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

  iterator<T> begin() noexcept {
    if (_cgh) {
      return iterator<T>(getAccessor(_cgh, *_buffer), 0, reinterpret_cast<ptrdiff_t>(this));
    }
    return iterator<T>(0, reinterpret_cast<ptrdiff_t>(this));
  }

  iterator<T> end() noexcept {
    if (_cgh) {
      return iterator<T>(getAccessor(_cgh, *_buffer), _size, reinterpret_cast<ptrdiff_t>(this));
    }
    return iterator<T>(_size, reinterpret_cast<ptrdiff_t>(this));
  }

  const iterator<T> begin() const noexcept {
    if (_cgh) {
      return iterator<T>(getAccessor(_cgh, *_buffer), 0, reinterpret_cast<ptrdiff_t>(this));
    }
    return iterator<T>(0, reinterpret_cast<ptrdiff_t>(this));
  }

  const iterator<T> end() const noexcept {
    if (_cgh) {
      return iterator<T>(getAccessor(_cgh, *_buffer), _size, reinterpret_cast<ptrdiff_t>(this));
    }
    return iterator<T>(_size, reinterpret_cast<ptrdiff_t>(this));
  }

  size_t size() const { return _size; }

  void resize(size_type size) {
    // TODO: implement it
  }

  operator typename std::remove_cv<T>::type() const {
    T tmp(_size);
    // download data to tmp
    // TODO: implement it to allow for a gvector to be asigned to an std::vector for
    // data transfer back to the host
    return tmp;
  }

  void swap(gvector &other) {
    std::swap(_buffer, other._buffer);
    std::swap(_size, other._size);
  }

  reference operator[](difference_type n) const {
    return getAccessor(_cgh, *_buffer)[n];
  }

protected:
  // an iterator has to register itself by its container
  // this is necessary to allow the container to update each iterator
  // as soon as the accessor is valid
  auto registerIterator(iterator<T>& it) {
    auto pair = std::make_pair(_iterators.size(), &it);
    _iterators.insert(pair);
    return pair.first;
  }

  // if a iterator is destroyed it tells its container to forget it
  void forgetIterator(size_t id){
    _iterators[id] = nullptr;
  }

private:
  // this is called by the gvector's base class gvector_base and is triggert after the
  // executor updates the gvectors cgh
  void updateAccessors() final {
    sycl_accessor_type acc;
    _cgh->require(*_buffer, acc);

    for (auto& p : _iterators) {
      if (p.second) {
        p.second->setAccessor(acc);
      }
    }
  }

  static sycl_accessor_type getAccessor(
      cl::sycl::handler* _cgh, cl::sycl::buffer<value_type> _buffer) {
    sycl_accessor_type accessor;
    _cgh->require(_buffer, accessor);
    return accessor;
  }

  std::unique_ptr<cl::sycl::buffer <value_type>> _buffer;
  std::map<size_t, iterator<T>*> _iterators;
  size_t _size;
};

template<typename T>
gvector<T> gpu_vector(T &vec) {
  return gvector<T>(vec);
}
}
}
