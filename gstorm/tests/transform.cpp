#include <gstorm.h>
#include <iostream>
#include <range/v3/all.hpp>
#include <vector>

using namespace ranges;
using namespace gstorm;

template<typename Rng>
void printRng(Rng&& rng) {
    RANGES_FOR(auto
    x, rng) { std::cout << x << " "; }
    std::cout << "\n";
}


void vadd() {
    size_t size = 10;

    std::vector<int> xs = view::iota(1) | view::take(size);
    std::vector<int> ys = view::repeat(1) | view::take(size);
    std::vector<int> out(size);

    // copy to gpu
    auto gpu_xs = xs | gpu::copy;
    auto gpu_ys = ys | gpu::copy;
    auto gpu_out = out | gpu::copy;

    auto add = [](auto&& tpl) { return std::get<0>(tpl) + std::get<1>(tpl); };
    auto gpu_zs = view::zip(gpu_xs, gpu_ys) | view::transform(add);

    auto id = [](auto&& in) { return in; };
    gpu::algorithm::transform(gpu_zs, gpu_out, id);

    out = gpu_out;

    auto zs = view::zip(xs, ys) | view::transform(add);

    printRng(zs);
    printRng(out);
}

int main() { vadd(); }
