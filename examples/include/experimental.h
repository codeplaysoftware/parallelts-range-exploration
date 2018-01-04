#pragma once

#include <gstorm.h>

// a parallel stl like interface
// TODO: make it use real std::executors instead of any type that fits
namespace std {
namespace experimental {
template<typename ExecT, typename InRng, typename OutRng, typename UnaryFunc>
void transform(ExecT &&exec, InRng &&in, OutRng &out, UnaryFunc func) {
  exec.transform(in, out, std::move(func));
}

template<typename ExecT, typename InRng>
auto copy(ExecT &&exec, InRng &&in) {
  return gstorm::gpu::copy(exec, in);
}

template<typename ExecT, typename InRng, typename T, typename BinaryFunc>
auto reduce(ExecT &&exec, InRng &&in, T init, BinaryFunc func) {
  return exec.reduce(in, std::move(init), std::move(func));
}

}// experimental
}// std
