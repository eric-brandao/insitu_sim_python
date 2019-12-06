#include "bemflush_pscat.h"

std::complex<float> bemflush_pscat(Eigen::RowVector3f &r_coord,
    Eigen::MatrixX4f &node_x, Eigen::MatrixX4f &node_y,
    Eigen::Matrix4Xf Nzeta, Eigen::VectorXf Nweights,
    float k0, std::complex<float> beta,
    Eigen::VectorXcf ps){
    // complex number
    std::complex<float> j1(0.0, 1.0);
    // Vector of receiver coordinates (for vectorized integration)
    Eigen::RowVectorXf x_coord = r_coord[0] * Eigen::RowVectorXf::Ones(Nzeta.cols());
    Eigen::RowVectorXf y_coord = r_coord[1] * Eigen::RowVectorXf::Ones(Nzeta.cols());
    Eigen::RowVectorXf z_coord = r_coord[2] * Eigen::RowVectorXf::Ones(Nzeta.cols());
    // Number of elements and jacobian
    int Nel = node_x.rows();
    float jacobian = (pow(node_x(1,0) - node_x(0,0), 2.0))/4.0;
    // Initialization
    Eigen::RowVectorXcf gfield = Eigen::RowVectorXcf::Zero(Nel);
    std::complex<float> p_scat = 0.0;
    // Loop through elements once
    for (int j = 0; j < Nel; j++){
        // Transform the coordinate system for integration between -1,1 and +1,+1
        Eigen::RowVector4f xnode = node_x.row(j);
        Eigen::RowVector4f ynode = node_y.row(j);
        Eigen::RowVectorXf xzeta = xnode * Nzeta;
        Eigen::RowVectorXf yzeta = ynode * Nzeta;
        // Calculate the distance from el center to transformed integration points
        Eigen::RowVectorXf r = ((x_coord - xzeta).array().pow(2.0)+
            (y_coord - yzeta).array().pow(2.0)+
            z_coord.array().pow(2.0)).array().pow(0.5);
        // Calculate green function
        Eigen::RowVectorXcf g = -j1 * k0 * beta *(
            (((-j1 * k0 * r.array()).exp()).array())/(4.0 * M_PI * r.array())) * jacobian;
        // Integrate
        gfield(j) = g * Nweights;
    }// end of loop
    p_scat = gfield * ps;
    return p_scat;
}