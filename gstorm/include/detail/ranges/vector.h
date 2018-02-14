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
struct giterator : public std::random_access_iterator_tag, public traits::range_forward_traits<T> {
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

  giterator(size_t offset = 0, ptrdiff_t ref_back = 0) : accessor(), owner(ref_back), it(offset), id_(size_t_max) {
#ifndef __SYCL_DEVICE_ONLY__
    if (owner)
      id_ = (reinterpret_cast<gvector<T>*>(owner))->registerIterator(*this); // this should never be executed on the device
#endif
  }

  ~giterator(){
#ifndef __SYCL_DEVICE_ONLY__
    if (owner && id_ != size_t_max)
      (reinterpret_cast<gvector<T>*>(owner))->forgetIterator(id_); // this should never be executed on the device
#endif
  }

  giterator(const giterator &other)
      : has_accessor(other.has_accessor),
        accessor(other.accessor),
        owner(other.owner),
        it(other.it),
        id_(size_t_max) {
#ifndef __SYCL_DEVICE_ONLY__
    if (owner)
      id_ = (reinterpret_cast<gvector<T> *>(owner))->registerIterator(*this);
#endif
  }

  bool hasAccessor() const { return has_accessor; }

  sycl_accessor_type getAccessor() const { return accessor; }

  void setAccessor(sycl_accessor_type acc) { has_accessor = true; accessor = acc; }

  explicit giterator(sycl_accessor_type acc, size_t offset = 0, ptrdiff_t ref_back = 0) :
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

  giterator &operator++() {
    ++it;
    return *this;
  }

  giterator operator++(int) {
    auto ip = it;
    ++it;
    return giterator(accessor, ip, owner);
  }

  giterator &operator--() {
    --it;
    return *this;
  }

  giterator operator--(int) {
    auto ip = it;
    --it;
    return giterator(accessor, ip, owner);
  }

  reference operator[](difference_type n) { return accessor[it + n]; }

  friend giterator advance(const giterator &lhs, difference_type n) {
    return lhs.advance(n);
  }

  giterator advance(difference_type n) {
    it += n;
    return *this;
  }

  friend giterator operator+(const giterator &lhs, difference_type n) {
    return giterator(lhs.accessor, lhs.it + n, lhs.owner);
  }

  friend giterator operator+(difference_type n, const giterator &rhs) {
    return giterator(rhs.accessor, rhs.it + n, rhs.owner);
  }

  friend giterator operator-(const giterator &lhs, difference_type n) {
    return giterator(lhs.accessor, lhs.it - n, lhs.owner);
  }

  friend difference_type operator-(const giterator &left, const giterator &right) {
    return left.it - right.it;
  }

  friend giterator &operator-=(giterator &lhs, difference_type n) {
    lhs.it -= n;
    return lhs;
  }

  friend giterator &operator+=(giterator &lhs, difference_type n) {
    lhs.it += n;
    return lhs;
  }

  friend bool operator<(const giterator &left, const giterator &right) {
    return left.it < right.it;
  }

  friend bool operator>(const giterator &left, const giterator &right) {
    return right < left;
  }

  friend bool operator<=(const giterator &left, const giterator &right) {
    return left < right || left == right;
  }

  friend bool operator>=(const giterator &left, const giterator &right) {
    return right < left || right == left;
  }

  bool operator==(const giterator &other) const { return other.it == it; }

  bool operator!=(const giterator &other) const { return !(other == *this); }

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

  using iterator = giterator<T>;
  using sentinel = giterator<T>;

  friend struct giterator<T>;

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

  giterator<T> begin() noexcept {
    if (_cgh) {
      return giterator<T>(getAccessor(_cgh, *_buffer), 0, reinterpret_cast<ptrdiff_t>(this));
    }
    return giterator<T>(0, reinterpret_cast<ptrdiff_t>(this));
  }

  giterator<T> end() noexcept {
    if (_cgh) {
      return giterator<T>(getAccessor(_cgh, *_buffer), _size, reinterpret_cast<ptrdiff_t>(this));
    }
    return giterator<T>(_size, reinterpret_cast<ptrdiff_t>(this));
  }

  const giterator<T> begin() const noexcept {
    if (_cgh) {
      return giterator<T>(getAccessor(_cgh, *_buffer), 0, reinterpret_cast<ptrdiff_t>(this));
    }
    return giterator<T>(0, reinterpret_cast<ptrdiff_t>(this));
  }

  const giterator<T> end() const noexcept {
    if (_cgh) {
      return giterator<T>(getAccessor(_cgh, *_buffer), _size, reinterpret_cast<ptrdiff_t>(this));
    }
    return giterator<T>(_size, reinterpret_cast<ptrdiff_t>(this));
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
  auto registerIterator(giterator<T>& it) {
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
  std::map<size_t, giterator<T>*> _iterators;
  size_t _size;
};

template<typename T>
gvector<T> gpu_vector(T &vec) {
  return gvector<T>(vec);
}
}
}
