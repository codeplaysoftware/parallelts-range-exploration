#include <chrono>
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <range/v3/all.hpp>

using namespace ranges;

template <typename Rng>
void printRng(Rng&& rng) {
  RANGES_FOR(auto x, rng)
  {
    std::cout << x << " ";
  }
  std::cout << "\n";
}

void vectorNegate() {
  auto size = 10u;
  std::vector<int> xs = view::iota(1) | view::take(size);

  auto neg = [](auto&& x) { return -x; };

  //std::vector<int> ys = xs | copy | action::transform(neg);
  auto ys = copy(xs);
  action::transform(ys, neg);

  std::cout << "xs: ";
  printRng(xs);

  std::cout << "ys: ";
  printRng(ys);
}

void vectorAddition() {
  auto size = 10u;
  auto xs = view::iota(1) | view::take(size);
  auto ys = view::iota(1) | view::take(size);

  auto add = [](auto&& xy) { return std::get<0>(xy) + std::get<1>(xy); };

  //   copying from a view doesn't work, but maybe it should ...
  //   this would require specifying the parituclar range type, e.g. std::vector
  // auto zs = view::zip(xs, ys) | copy | action::transform(add);
  auto zs = view::zip(xs, ys) | view::transform(add);

  std::cout << "zs: ";
  printRng(zs);
}

void saxpy() {
  // TODO: generalize these two helper functions
  // auto asIntVector = make_pipeable([](auto&& rng) -> std::vector<int> {
  //   return rng;
  // });

  auto applyOnTuple = [](auto&& f) {
    return [f = std::move(f)](auto&& tpl) {
      return f(std::get<0>(tpl), std::get<1>(tpl), std::get<2>(tpl));
    };
  };

  auto size = 10u;
  auto xs = view::iota(1) | view::take(size);
  auto ys = view::iota(2) | view::take(size);
  auto sa = view::repeat(5);

  auto saxpy = [](auto a, auto x, auto y) {
    return a * x + y;
  };

  std::vector<int> zs = view::zip(sa, xs, ys) | view::transform(applyOnTuple(saxpy));// | asIntVector;

  std::cout << "zs: ";
  printRng(zs);
}

// for comparison with thrust ...
// is this as efficient as the first saxpy function?
void saxpy2() {
  // TODO: generalize this helper functions
  auto applyOnTuple = [](auto&& f) {
    return [f = std::move(f)](auto&& tpl) {
      return f(std::get<0>(tpl), std::get<1>(tpl));
    };
  };

  auto size = 10u;
  auto xs = view::iota(1) | view::take(size);
  auto ys = view::iota(2) | view::take(size);
  auto sa = view::repeat(5);

  auto mult = [](auto a, auto x) { return a * x; };
  auto plus = [](auto x, auto y) { return x + y; };

  auto saxs = view::zip(sa, xs) | view::transform(applyOnTuple(mult));
  std::vector<int> zs = view::zip(saxs, ys) | view::transform(applyOnTuple(plus)); //| asIntVector;

  std::cout << "zs: ";
  printRng(zs);
}

void norm() {
  auto square = [](auto x) { return x * x; };
  auto plus = [](auto x, auto y) { return x + y; };

  auto xs = view::iota(1) | view::take(10);

  auto squared = xs | view::transform(square);
  auto n = std::sqrt( accumulate(squared, 0, plus) );
  std::cout << "norm: " << n << "\n";
}

void dotProduct() {
  auto dot = [](auto&& xs, auto&& ys) {
    auto mult = [](auto&& xy) { return std::get<0>(xy) * std::get<1>(xy);  };
    auto plus = [](auto x, auto y) { return x + y; };

    return accumulate(view::zip(xs, ys) | view::transform(mult), 0, plus);
  };

  auto size = 10u;
  auto xs = view::iota(1) | view::take(size);
  auto ys = view::iota(2) | view::take(size);

  auto res = dot(xs, ys);

  // I would like to write(, but I'm not allowed):
  //auto res = view::zip(xs, ys) | view::transform(mult) | accumulate(0, plus);

  std::cout << "res: " << res << "\n";
}

void monte_carlo() {
  auto estimate_pi = [](auto i) -> float {
    // do stuff ... need a random number generator on the GPU ...
    return static_cast<float>(i);
  };
  auto plus = [](auto x, auto y) { return x + y; };

  auto M = 100;
  auto estimate = accumulate(view::ints(0, M) | view::transform(estimate_pi), 0.0f, plus);
  estimate /=  M;

  std::cout << "M: " << M << "\n";
}

int main()
{
  vectorNegate();
  vectorAddition();
  saxpy();
  saxpy2();
  norm();
  dotProduct();
  monte_carlo();
}
