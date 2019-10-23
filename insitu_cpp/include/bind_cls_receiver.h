#ifndef BIND_CLS_RECEIVER_H
#define BIND_CLS_RECEIVER_H

#include <iostream>

#include "pybind11/complex.h"
#include "pybind11/eigen.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "unsupported/Eigen/Polynomials"
#include "receiver.h"

namespace py = pybind11;

void bind_cls_receivercpp(py::module &m);
#endif /* BIND_CLS_RECEIVER_H */