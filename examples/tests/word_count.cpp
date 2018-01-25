#include "gtest/gtest.h"

#include <iostream>
#include <string>

#include <gstorm.h>
#include <range/v3/all.hpp>

#include "experimental.h"

struct String : public testing::Test {};

// determines whether the character is alphabetical
bool is_alpha(const char c) { return (c >= 'A' && c <= 'z'); }

// determines whether the right character begins a new word
struct is_word_start {
  constexpr is_word_start(){};

  template <typename T>
  bool operator()(const T& tpl) const {
    auto left = std::get<0>(tpl);
    auto right = std::get<1>(tpl);
    return is_alpha(right) && !is_alpha(left);
  }
};

TEST_F(String, WordCount) {

  std::string text =
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod "
      "tempor incididunt ut labore et dolore magna aliqua. Tortor id aliquet "
      "lectus proin nibh. Posuere sollicitudin aliquam ultrices sagittis. Enim "
      "blandit volutpat maecenas volutpat blandit aliquam etiam erat. "
      "Tristique magna sit amet purus gravida quis blandit turpis. Lectus urna "
      "duis convallis convallis tellus. Lacus sed turpis tincidunt id. Eu "
      "consequat ac felis donec et odio pellentesque diam volutpat. Nunc sed "
      "id semper risus in hendrerit gravida rutrum quisque. Elementum nisi "
      "quis eleifend quam adipiscing vitae proin. Orci ac auctor augue mauris "
      "augue neque gravida in.";

  {
    gstorm::sycl_exec exec;

    auto gtext = std::experimental::copy(exec, text);

    auto init = gtext
              | ranges::view::transform([](auto a) { return a; } )
              | ranges::view::take(text.size() - 1);
    auto tail = gtext
              | ranges::view::transform([](auto a) { return a; } )
              | ranges::view::drop(1);

    auto word_starts = ranges::view::zip(init, tail)
                     | ranges::view::transform(is_word_start{});

    auto result = std::experimental::reduce(exec, word_starts, 0, std::plus<int>{});

    auto expected = ranges::accumulate(
                      ranges::view::zip(
                        text | ranges::view::take(text.size() - 1),
                        text | ranges::view::drop(1))
                      | ranges::view::transform(is_word_start{}),
                      0.0f, std::plus<int>{});

    EXPECT_EQ(expected, result);
  }

}
