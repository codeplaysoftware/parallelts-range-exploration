//
// Created by m_haid02 on 03.05.17.
//

#pragma once

#include <CL/sycl.hpp>

namespace gstorm {

class sycl_exec;

namespace range {

struct gvector_base {
  gvector_base() : _cgh(nullptr), _id(-1) {}
  virtual ~gvector_base() {}
  void setCGH(cl::sycl::handler &cgh) {
    _cgh = &cgh;
    updateAccessors();
  }

  void setExecutorPtr(gstorm::sycl_exec *ptr) {
    _exec = ptr;
  }

  void setID(size_t id) { _id = id; }

  virtual void updateAccessors() = 0;

protected:
  cl::sycl::handler *_cgh;
  gstorm::sycl_exec* _exec;
  size_t _id;
};
}
}