#include "bind_fun_bemflush_mtx2.h"

void bind_bemflush_mtx2(py::module &m)
{
    m.def("_bemflush_mtx2", bemflush_mtx2,
    "Assembles the BEM matrix coefficients",
    py::arg("Nweights").noconvert(),
    py::arg("r_mtx").noconvert(),
    py::arg("jacobian"),
    py::arg("k0"), py::arg("beta")
    );
}