#ifndef BIND_FUN_BEMFLUSH_UXSCAT_H
#define BIND_FUN_BEMFLUSH_UXSCAT_H

#include <iostream>

#include "pybind11/complex.h"
#include "pybind11/eigen.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "unsupported/Eigen/Polynomials"
#include "bemflush_uxscat.h"

namespace py = pybind11;

void bind_bemflush_uxscat(py::module &m);

#endif /* BIND_FUNBEMFLUSH_UXSCAT_H */