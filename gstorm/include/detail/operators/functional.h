#pragma once

namespace gstorm {
  namespace functional {
//    struct plus {
//      template<typename T>
//      auto operator()(T&& tpl) const {
//        return std::get<0>(tpl) + std::get<1>(tpl);
//      }
//
//      template<typename T>
//      auto operator()(T&& x, T&& y) const { return x + y; }
//    };
//
//    struct minus {
//      template<typename T>
//      auto operator()(T&& tpl) const {
//        return std::get<0>(tpl) - std::get<1>(tpl);
//      }
//
//      template<typename T>
//      auto operator()(T&& x, T&& y) const { return x - y; }
//    };
//
//    struct multiplies {
//      template<typename T>
//      auto operator()(T&& tpl) const {
//        return std::get<0>(tpl) * std::get<1>(tpl);
//      }
//
//      template<typename T>
//      auto operator()(T&& x, T&& y) const { return x * y; }
//    };

    struct identity {
      template<typename T>
      auto operator()(T&& value) -> T const {
        return value;
      }
    };
  }
}