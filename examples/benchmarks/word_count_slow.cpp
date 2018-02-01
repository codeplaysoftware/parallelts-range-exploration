#include <chrono>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>

#include <gstorm.h>
#include <CL/sycl.hpp>
#include <range/v3/all.hpp>

#include "aligned_allocator.h"
#include "experimental.h"

// determines whether the character is alphabetical
bool is_alpha(const char c) { return (c >= 'A' && c <= 'z'); }

// determines whether the right character begins a new word
struct is_word_start {
  constexpr is_word_start(){};

  template <typename T>
  int operator()(const T& tpl) const {
    auto left = std::get<0>(tpl);
    auto right = std::get<1>(tpl);
    return is_alpha(right) && !is_alpha(left);
  }
};

int main(int argc, char* argv[]) {
  const size_t base_size = 1024 * 128;

  size_t multiplier = 1;

  if (argc > 1) {
    std::stringstream{argv[1]} >> multiplier;
  }

  std::cout << "Size: " << multiplier << "\n";
  const auto vsize = base_size * multiplier;
  const auto iterations = 100;

  std::basic_string<char, std::char_traits<char>, aligned_allocator<char,4096>> text(vsize, 'a');

  // Create some "words" by adding spaces
  for (auto i = 4ul; i < vsize; i += 15) {
    text[i] = ' ';
  }

  int result = 0;

  std::vector<int, aligned_allocator<int, 4096>> word_starts(vsize-1);
  std::vector<double> times{};

  cl::sycl::gpu_selector device_selector;
  auto q = cl::sycl::queue(device_selector,
                           {cl::sycl::property::queue::enable_profiling{}});
  std::cout << "Using device: "
            << q.get_device().get_info<cl::sycl::info::device::name>()
            << ", from: "
            << q.get_device()
                .get_platform()
                .get_info<cl::sycl::info::platform::name>()
            << "\n";

  for (auto i = 0; i < iterations; ++i) {
    auto start = std::chrono::system_clock::now();
    {
      gstorm::sycl_exec exec(q);
      auto gtext = std::experimental::copy(exec, text);
      auto gstarts = std::experimental::copy(exec, word_starts);

      auto init = gtext
                | ranges::view::transform([](auto a) { return a; } )
                | ranges::view::take(text.size() - 1);
      auto tail = gtext
                | ranges::view::transform([](auto a) { return a; } )
                | ranges::view::drop(1);

      auto zipped = ranges::view::zip(init, tail);
      std::experimental::transform(exec, zipped, gstarts, is_word_start{});

      result = std::experimental::reduce(exec, gstarts, 0, std::plus<int>{});

    }
    auto end = std::chrono::system_clock::now();

    auto time_taken =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    times.push_back(time_taken.count() / 1000.0);
    std::cout << "\r" << (i + 1) << "/" << iterations << std::flush;
  }
  std::cout << "\n";

  ranges::sort(times);
  std::cout << "Median time: " << times[iterations / 2] << " ms\n";

  auto expected = ranges::accumulate(
                    ranges::view::zip(
                      text | ranges::view::take(text.size() - 1),
                      text | ranges::view::drop(1))
                    | ranges::view::transform(is_word_start{}),
                    0.0f, std::plus<int>{});

  if (expected != result) {
    std::cout << "Mismatch between expected and actual result!\n";
    return 1;
  }
}
