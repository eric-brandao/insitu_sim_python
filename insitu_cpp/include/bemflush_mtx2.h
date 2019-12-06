#ifndef BEMFLUSH_MTX2_H
#define BEMFLUSH_MTX2_H

#include <iostream>
#include "pybind11/pybind11.h"
#include "pybind11/eigen.h"
#include "pybind11/numpy.h"
#include <complex>


Eigen::MatrixXcf bemflush_mtx2(Eigen::VectorXf Nweights,
    Eigen::MatrixXf r_mtx, float jacobian,
    float k0, std::complex<float> beta);
#endif /* BEMFLUSH_MTX2_H */