#ifndef BIND_CLS_SOURCE_H
#define BIND_CLS_SOURCE_H

#include <iostream>

#include "pybind11/complex.h"
#include "pybind11/eigen.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "unsupported/Eigen/Polynomials"
#include "source.h"
// #include "ray.h"

namespace py = pybind11;

void bind_cls_sourcecpp(py::module &m);
#endif /* BIND_CLS_SOURCE_H */