#include "bind_cls_receiver.h"

void bind_cls_receivercpp(py::module &m){
    py::class_<Receivercpp>(m, "Receivercpp")
        .def(py::init<
        Eigen::RowVector3f,
        Eigen::RowVector3f
        >())
        .def("point_to_source", &Receivercpp::point_to_source)
        .def_readwrite("coord", &Receivercpp::coord)
        .def_readwrite("orientation", &Receivercpp::orientation);
}