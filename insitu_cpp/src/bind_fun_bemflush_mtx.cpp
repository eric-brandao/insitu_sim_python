#include "bind_fun_bemflush_mtx.h"

void bind_bemflush_mtx(py::module &m)
{
    m.def("_bemflush_mtx", bemflush_mtx,
    "Assembles the BEM matrix coefficients",
    py::arg("el_center").noconvert(),
    py::arg("node_x").noconvert(),
    py::arg("node_y").noconvert(),
    py::arg("Nzeta").noconvert(),
    py::arg("Nweights").noconvert(),
    py::arg("k0"), py::arg("beta")
    );
}