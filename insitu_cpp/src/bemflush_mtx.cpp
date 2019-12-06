#include "bemflush_mtx.h"

Eigen::MatrixXcf bemflush_mtx(Eigen::MatrixX2f &el_center,
    Eigen::MatrixX4f &node_x, Eigen::MatrixX4f &node_y,
    Eigen::Matrix4Xf Nzeta, Eigen::VectorXf Nweights,
    float k0, std::complex<float> beta){

    std::complex<float> j1(0.0, 1.0);
    // std::complex<float> jk0 = j1 * k0;
    int Nel = el_center.rows();
    // float jacobian = ((el_center[1,0] - el_center[0,0]).pow(2.0))/4.0;
    float jacobian = (pow(el_center(1,0) - el_center(0,0), 2.0))/4.0;
    // std::cout << "The number of elements is: " << Nel << std::endl;
    Eigen::MatrixXcf bem_mtx = Eigen::MatrixXcf::Zero(Nel, Nel);
    // Loop through elements twice
    for (int i = 0; i < Nel; i++){
        Eigen::RowVector2f xy_center = el_center.row(i);
        Eigen::RowVectorXf x_center = xy_center[0] * Eigen::RowVectorXf::Ones(Nzeta.cols());
        Eigen::RowVectorXf y_center = xy_center[1] * Eigen::RowVectorXf::Ones(Nzeta.cols());
        // std::cout << "xy_center: " << xy_center << std::endl;
        // std::cout << x_center << std::endl;
        for (int j = 0; j < Nel; j++){
            if(i <= j){
                // Transform the coordinate system for integration between -1,1 and +1,+1
                Eigen::RowVector4f xnode = node_x.row(j);
                Eigen::RowVector4f ynode = node_y.row(j);
                Eigen::RowVectorXf xzeta = xnode * Nzeta;
                // std::cout << "size of xzeta: " << xzeta.size() << std::endl;
                // std::cout << "size of xzeta: " << x_center.size() << std::endl;
                // std::cout << "x_node: " << xnode << std::endl;
                // std::cout << "    " <<  std::endl;
                // std::cout << xzeta << std::endl;
                Eigen::RowVectorXf yzeta = ynode * Nzeta;
                // Calculate the distance from el center to transformed integration points
                // Eigen::RowVectorXf r = (x_center - xzeta);
                Eigen::RowVectorXf r = ((x_center - xzeta).array().pow(2.0)+
                    (y_center - yzeta).array().pow(2.0)).array().pow(0.5);
                // Eigen::RowVectorXcf g = j1 * k0 * beta * Eigen::RowVectorXcf::Ones(r.cols());
                Eigen::RowVectorXcf g = j1 * k0 * beta *(
                    (((-j1 * k0 * r.array()).exp()).array())/(4.0 * M_PI * r.array())) * jacobian;
                // std::cout << "beta: " << beta << std::endl;
                // std::cout << g << std::endl;
                bem_mtx(i,j) = g * Nweights;
                // g = j1 * k0 * beta
                // Eigen::RowVectorXcf g(r.size());
                // g = k0 * r;
            }
            else{
                bem_mtx(i,j) = bem_mtx(j,i);
            }
        }// end of first loop
    }//end of second loop
    return bem_mtx;
}