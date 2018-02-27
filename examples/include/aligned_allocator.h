#ifndef ALIGNED_ALLOCATOR_H
#define ALIGNED_ALLOCATOR_H

#include <cstddef>
#include <new>

#include <stdlib.h>

template <class T, std::size_t Alignment = alignof(T)>
struct aligned_allocator {
  using value_type = T;

  template <class U>
  struct rebind {
    using other = aligned_allocator<U, Alignment>;
  };

  aligned_allocator() = default;

  template <class U>
  constexpr aligned_allocator(const aligned_allocator<U, Alignment>&) noexcept {
  }

  T* allocate(std::size_t n) {
    if (n > std::size_t(-1) / sizeof(T)) {
      throw std::bad_alloc{};
    }

    auto p = static_cast<T*>(aligned_alloc(Alignment, n * sizeof(T)));
    if (p) {
      return p;
    }

    throw std::bad_alloc{};
  }

  void deallocate(T* p, std::size_t) noexcept { free(p); }

  bool operator==(const aligned_allocator&) { return true; }

  bool operator!=(const aligned_allocator& rhs) { return !operator==(rhs); }
};

#endif // ALIGNED_ALLOCATOR_H
