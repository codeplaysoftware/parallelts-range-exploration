#include "gtest/gtest.h"

#include <gstorm.h>
#include <vector>
#include <iostream>
#include <range/v3/all.hpp>

#include <CL/sycl.hpp>

#include "experimental.h"

struct Accessor : public testing::Test {};

namespace tryout
{
using sycl_accessor_type = cl::sycl::accessor<int, 1,
                                              cl::sycl::access::mode::read_write,
                                              cl::sycl::access::target::global_buffer,
                                              cl::sycl::access::placeholder::true_t>;

  struct iterator{
    auto get() const { return x; }
    sycl_accessor_type x;
    ptrdiff_t owner;
    size_t it;
    size_t _id;
  };

  template <typename T> struct box
  {
    T elem;

    constexpr T const &get() const & noexcept
    {
      return elem;
    }
  };

  template <typename T> struct adapt : public box<T> {
    auto foo() const { return this->get().get(); }
  };

}

TEST_F(Accessor, TestAccessor) {

  cl::sycl::default_selector device_selector;
  cl::sycl::queue comQueue(device_selector);

  tryout::adapt<tryout::iterator> x;
  cl::sycl::nd_range<1> config{1, cl::sycl::range < 1 > {1}};
  std::vector<int> a = { 1 };
  cl::sycl::buffer<int,1> a_sycl(a.data(), cl::sycl::range<1>(1));
  static_assert(std::is_standard_layout<decltype(x)>::value, "not a standard layout type");

  tryout::sycl_accessor_type a_acc;

  comQueue.submit([&](cl::sycl::handler& cgh) {
    cgh.require(a_sycl, a_acc);
    x.elem.x = a_acc;

    cgh.template parallel_for<class myKernel>(config, [=](cl::sycl::nd_item<1> id) {
      x.foo();
    });

  });
}
