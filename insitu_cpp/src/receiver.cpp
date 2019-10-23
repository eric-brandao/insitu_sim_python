#include "receiver.h"

// Method to point receiver to a given sound source
Eigen::RowVector3f Receivercpp::point_to_source(
    Eigen::RowVector3f &source_coord){
    orientation = coord - source_coord;
    orientation = orientation / orientation.norm();
    return orientation;
}