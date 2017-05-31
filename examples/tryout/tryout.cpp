//
// Created by m_haid02 on 01.05.17.
//

#include <gstorm.h>
#include <vector>
#include <iostream>
#include <memory>
#include <range/v3/all.hpp>
#include <type_traits>

#include <CL/sycl.hpp>
#include <meta/optional>

using namespace gstorm;
using namespace cl::sycl;
using namespace ranges::v3;


// a parallel stl like interface
// TODO: make it use real std::executors instead of any type that fits
namespace std {
namespace experimental {
template<typename ExecT, typename InRng, typename OutRng, typename UnaryFunc>
void transform(ExecT &&exec, const InRng &in, OutRng &out, UnaryFunc func) {
  exec.transform(in, out, func);
}

template<typename ExecT, typename InRng>
auto copy(ExecT &&exec, InRng &in) {
  return gpu::copy(exec, in);
}

template<typename ExecT, typename InRng, typename T, typename BinaryFunc>
auto reduce(ExecT &&exec, InRng &in, T init, BinaryFunc func) {
  return exec.reduce(in, init, func);
}

}// experimental
}// std



namespace tryout
{
using sycl_accessor_type = cl::sycl::accessor<int, 1,
                                              cl::sycl::access::mode::read_write,
                                              cl::sycl::access::target::global_buffer>;

  struct iterator{
    auto get() const { return x; }
  private:
    mutable std::experimental::optional<sycl_accessor_type> x;
    mutable ptrdiff_t owner;
    mutable size_t it;
    mutable size_t _id;
  };

  template <typename T> struct box
  {
  private:
    T elem;
  public:
    constexpr T const &get() const & noexcept
    {
      return elem;
    }
  };

  template <typename T> struct adapt : private box<T> {
    auto foo() const { return this->get().get().value(); }
  };

}


class myKernel;

int main() {


  cl::sycl::queue comQueue;

  tryout::adapt<tryout::iterator> x;
  cl::sycl::nd_range<1> config{1, cl::sycl::range < 1 > {1}};

  static_assert(std::is_standard_layout<decltype(x)>::value, "not a standard layout type");

  comQueue.submit([&](cl::sycl::handler& cgh){


    cgh.template parallel_for< class myKernel>(config, [=](cl::sycl::nd_item<1> id) {

      x.foo();
    });

  });


  size_t vsize = 1024;

  std::vector<int> va(vsize, 1);
  std::vector<int> vb(vsize, 2);
  std::vector<int> vc(vsize, 3);
  std::vector<int> vd(vsize, 0);

  int v = 0;
  int v2 = 0;
  {
    sycl_exec exec;

    auto ga = std::experimental::copy(exec, va);
    auto gb = std::experimental::copy(exec, vb);
    auto gc = std::experimental::copy(exec, vc);
    auto gd = std::experimental::copy(exec, vd);

    static_assert(std::is_standard_layout<decltype(std::declval<decltype(gc)>().begin())>::value && "not an STLT");

    auto fma = [=](auto tpl) { return std::get<0>(tpl) * std::get<1>(tpl) + std::get<2>(tpl); };

    static_assert(std::is_standard_layout<std::reference_wrapper<decltype(fma)>>::value, "baem");


    auto zip = view::zip(ga, gb, gc);
    // the slow path through a transform and a reduction with a temporary vector gd
    std::experimental::transform(exec, zip, gd, fma);
    v = std::experimental::reduce(exec, gd, 0, [](auto a, auto b) { return a + b; });
    auto ziped = view::zip(ga, gb, gc);

    auto fma_ziped = view::transform(ziped, fma);
    v2 = std::experimental::reduce(exec, fma_ziped, 0, [](auto a, auto b) { return a + b; });
  }

  std::cout << "results: slow path: " << v << " fast path: " << v2 <<" gold is: " << std::accumulate(vd.begin(), vd.end(), 0, std::plus<>()) << std::endl;


}
