#ifndef BIND_FUN_BEMFLUSH_RMTX_H
#define BIND_FUN_BEMFLUSH_RMTX_H

#include <iostream>

#include "pybind11/complex.h"
#include "pybind11/eigen.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "unsupported/Eigen/Polynomials"
#include "bemflush_rmtx.h"

namespace py = pybind11;

void bind_bemflush_rmtx(py::module &m);

#endif /* BIND_FUNBEMFLUSH_RMTX_H */