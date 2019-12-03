#ifndef BEMFLUSH_MTX_H
#define BEMFLUSH_MTX_H

#include <iostream>
#include <iostream>
#include "pybind11/pybind11.h"
#include "pybind11/eigen.h"
#include "pybind11/numpy.h"
#include <complex>


Eigen::MatrixXcf bemflush_mtx(Eigen::MatrixX2f &el_center,
    Eigen::MatrixX4f &node_x, Eigen::MatrixX4f &node_y,
    Eigen::Matrix4Xf Nzeta, Eigen::VectorXf Nweights,
    float k0, std::complex<float> beta);
#endif /* BEMFLUSH_MTX_H */