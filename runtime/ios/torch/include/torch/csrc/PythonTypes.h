#pragma once

#include <torch/csrc/python_headers.h>
#include <torch/csrc/Types.h>

namespace torch {

struct THPVoidTensor {
  PyObject_HEAD
  THVoidTensor *cdata;
  char device_type;
  char data_type;
};

struct THPVoidStorage {
  PyObject_HEAD
  THVoidStorage *cdata;
};

} // namespace torch
