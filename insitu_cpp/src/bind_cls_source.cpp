#include "bind_cls_source.h"
// #include "bind_cls_ray.h"

void bind_cls_sourcecpp(py::module &m){
    py::class_<Sourcecpp>(m, "Sourcecpp")
        .def(py::init<
        Eigen::RowVector3f,
        Eigen::RowVector3f,
        Eigen::RowVectorXf,
        Eigen::RowVectorXf,
        Eigen::RowVectorXf,
        double
        // std::vector<Sourcecpp> &rays
        >())
        .def_readwrite("coord", &Sourcecpp::coord)
        .def_readwrite("orientation", &Sourcecpp::orientation)
        .def_readwrite("power_dB", &Sourcecpp::power_dB)
        .def_readwrite("eq_dB", &Sourcecpp::eq_dB)
        .def_readwrite("power_lin", &Sourcecpp::power_lin)
        .def_readwrite("delay", &Sourcecpp::delay);
        // .def_readwrite("rays", &Sourcecpp::rays);
}