#include "bemflush_mtx2.h"

Eigen::MatrixXcf bemflush_mtx2(Eigen::VectorXf Nweights,
    Eigen::MatrixXf r_mtx, float jacobian,
    float k0, std::complex<float> beta){
    // complex number and Nel
    std::complex<float> j1(0.0, 1.0);
    int Nel = int(pow(r_mtx.rows(), 0.5));
    Eigen::MatrixXcf bem_mtx = Eigen::MatrixXcf::Zero(Nel, Nel);
    // set counter
    int count = 0;
    // Loop through elements twice
    for (int i = 0; i < Nel; i++){
        for (int j = 0; j < Nel; j++){
            // std::cout << r_mtx.row(count) << std::endl;
            if(i <= j){
                Eigen::RowVectorXf r = r_mtx.row(count);
                Eigen::RowVectorXcf g = j1 * k0 * beta *(
                    (((-j1 * k0 * r.array()).exp()).array())/(4.0 * M_PI * r.array())) * jacobian;
                bem_mtx(i,j) = g * Nweights;
            }
            else{
                bem_mtx(i,j) = bem_mtx(j,i);
            }
            count++;
        }// end of first loop
    }//end of second loop
    return bem_mtx;
}