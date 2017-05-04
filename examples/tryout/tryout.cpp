//
// Created by m_haid02 on 01.05.17.
//

#include <gstorm.h>
#include <vector>
#include <iostream>
#include <memory>
#include <range/v3/all.hpp>
#include "../../../computeCpp/include/CL/sycl.hpp"


using namespace gstorm;
using namespace cl::sycl;
using namespace ranges::v3;

namespace std {
namespace experimental {
template<typename ExecT, typename InRng, typename OutRng, typename UnaryFunc>
void transform(ExecT &&exec, const InRng &in, OutRng &out, UnaryFunc func) {
  exec.transform(in, out, func);
}
}
}

int main() {
  std::vector<int> va(128, 1);
  std::vector<int> vb(128, 2);
  std::vector<int> vc(128, 3);
  std::vector<int> vd(128, 0);

  {
    sycl_exec exec;

    auto ga = gpu::copy(exec, va);
    auto gb = gpu::copy(exec, vb);
    auto gc = gpu::copy(exec, vc);
    auto gd = gpu::copy(exec, vd);


    auto t2 = [=](auto v) { return v*2; };

    auto fma = [=](auto tpl) { return std::get<0>(tpl) * std::get<1>(tpl) + std::get<2>(tpl); };

    auto zip = view::zip(ga, gb, gc);

    std::experimental::transform(exec, zip, gd, fma);
  }
  
  for(auto v : vd)
    std::cout << v << " ";
  std::cout << "\n";


}
