#include "bemflush_rmtx.h"

Eigen::MatrixXf bemflush_rmtx(Eigen::MatrixX2f &el_center,
    Eigen::MatrixX4f &node_x, Eigen::MatrixX4f &node_y,
    Eigen::Matrix4Xf Nzeta){
    // Number of elements and jacobian
    int Nel = el_center.rows();
    // float jacobian = (pow(el_center(1,0) - el_center(0,0), 2.0))/4.0;
    // initialize r matrix (distantces) - Nel^2 x Ngaussweigths^2
    Eigen::MatrixXf r_mtx = Eigen::MatrixXf::Zero(int(pow(Nel,2)), Nzeta.cols());
    // set counter
    int count = 0;
    // Loop through elements twice
    for (int i = 0; i < Nel; i++){
        // x, y coordinates of element's center (as a vector)
        Eigen::RowVector2f xy_center = el_center.row(i);
        Eigen::RowVectorXf x_center = xy_center[0] * Eigen::RowVectorXf::Ones(Nzeta.cols());
        Eigen::RowVectorXf y_center = xy_center[1] * Eigen::RowVectorXf::Ones(Nzeta.cols());
        for (int j = 0; j < Nel; j++){
            // Transform the coordinate system for integration between -1,1 and +1,+1
            Eigen::RowVector4f xnode = node_x.row(j);
            Eigen::RowVector4f ynode = node_y.row(j);
            Eigen::RowVectorXf xzeta = xnode * Nzeta;
            Eigen::RowVectorXf yzeta = ynode * Nzeta;
            // Calculate the distance from el center to transformed integration points
            r_mtx.row(count) = ((x_center - xzeta).array().pow(2.0)+
                (y_center - yzeta).array().pow(2.0)).array().pow(0.5);
            // increase counter
            count++;
        }// end of first loop
    }//end of second loop
    return r_mtx;
}