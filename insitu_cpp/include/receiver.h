#ifndef RECEIVER_H
#define RECEIVER_H

#include <iostream>
#include "pybind11/pybind11.h"
#include "pybind11/eigen.h"
#include "pybind11/numpy.h"

/* The class Raycpp is used to construct a list of
receiver objects with several properties. This will be used
during the ray tracing in several sub-functions*/

class Receivercpp
{
public:
    Receivercpp(
        Eigen::RowVector3f coord,
        Eigen::RowVector3f orientation
        ):
    coord(coord), orientation(orientation)
    {}
    ~Receivercpp()  {} // class destructor - can be automatic later
    // Method to point receiver to a given sound source
    Eigen::RowVector3f point_to_source(Eigen::RowVector3f &source_coord);
// Parameters of the Receivercpp class
Eigen::RowVector3f coord;
Eigen::RowVector3f orientation;
};
#endif /* RECEIVER_H */