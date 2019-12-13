#ifndef BEMFLUSH_UZSCAT_H
#define BEMFLUSH_UZSCAT_H

#include <iostream>
#include "pybind11/pybind11.h"
#include "pybind11/eigen.h"
#include "pybind11/numpy.h"
#include <complex>


std::complex<float> bemflush_uzscat(Eigen::RowVector3f &r_coord,
    Eigen::MatrixX4f &node_x, Eigen::MatrixX4f &node_y,
    Eigen::Matrix4Xf Nzeta, Eigen::VectorXf Nweights,
    float k0, std::complex<float> beta,
    Eigen::VectorXcf ps);
#endif /* BEMFLUSH_UZSCAT_H */