#ifndef BEMFLUSH_RMTX_H
#define BEMFLUSH_RMTX_H

#include <iostream>
#include "pybind11/pybind11.h"
#include "pybind11/eigen.h"
#include "pybind11/numpy.h"
// #include <complex>


Eigen::MatrixXf bemflush_rmtx(Eigen::MatrixX2f &el_center,
    Eigen::MatrixX4f &node_x, Eigen::MatrixX4f &node_y,
    Eigen::Matrix4Xf Nzeta);
#endif /* BEMFLUSH_RMTX_H */